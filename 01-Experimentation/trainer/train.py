

# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text  # A dependency of the preprocessing model
import tensorflow_addons as tfa

from absl import app
from absl import flags
from absl import logging

from official.nlp import optimization

TFHUB_HANDLE_ENCODER = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
TFHUB_HANDLE_PREPROCESS = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
TFDS_NAME = 'glue/cola' 
NUM_CLASSES = 2
SENTENCE_FEATURE = 'sentence'

LOCAL_MODEL_DIR = '/tmp/saved_model'
LOCAL_TB_DIR = '/tmp/logs'
LOCAL_CHECKPOINT_DIR = '/tmp/checkpoints'

FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 2, 'Nubmer of epochs')
flags.DEFINE_integer('per_replica_batch_size', 16, 'Per replica batch size')
flags.DEFINE_float('init_lr', 2e-5, 'Initial learning rate')
flags.DEFINE_float('dropout_ratio', 0.1, 'Dropout ratio')


def make_bert_preprocess_model(sentence_features, seq_length=128):
    """Returns a model mapping string features to BERT inputs."""

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(TFHUB_HANDLE_PREPROCESS)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=seq_length),
                            name='packer')
    model_inputs = packer(segments)
    return tf.keras.Model(input_segments, model_inputs)


def build_classifier_model(num_classes, dropout_ratio):
    """Creates a text classification model based on BERT encoder."""

    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )

    encoder = hub.KerasLayer(TFHUB_HANDLE_ENCODER, trainable=True, name='encoder')
    net = encoder(inputs)['pooled_output']
    net = tf.keras.layers.Dropout(rate=dropout_ratio)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model(inputs, net, name='prediction')


def get_data_pipeline(in_memory_ds, info, split, 
                      batch_size,  bert_preprocess_model):
    """Creates a sentence preprocessing pipeline."""
    
    is_training = split.startswith('train')
    dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[split])
    num_examples = info.splits[split].num_examples

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, num_examples


def set_job_dirs():
    """Sets job directories based on env variables set by Vertex AI."""
    
    model_dir = os.getenv('AIP_MODEL_DIR', LOCAL_MODEL_DIR)
    tb_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR', LOCAL_TB_DIR)
    checkpoint_dir = os.getenv('AIP_CHECKPOINT_DIR', LOCAL_CHECKPOINT_DIR)
    
    return model_dir, tb_dir, checkpoint_dir
    

def main(argv):
    """Starts a training run."""
    
    del argv
    logging.info('Setting up training.')
    logging.info('   epochs: {}'.format(FLAGS.epochs))
    logging.info('   per_replica_batch_size: {}'.format(FLAGS.per_replica_batch_size))
    logging.info('   init_lr: {}'.format(FLAGS.init_lr))
    logging.info('   dropout_ratio: {}'.format(FLAGS.dropout_ratio))

    # Set distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    
    global_batch_size = (strategy.num_replicas_in_sync *
                         FLAGS.per_replica_batch_size)
    
    # Configure input data pipelines
    tfds_info = tfds.builder(TFDS_NAME).info
    num_classes = tfds_info.features['label'].num_classes
    num_examples = tfds_info.splits.total_num_examples
    available_splits = list(tfds_info.splits.keys())
    labels_names = tfds_info.features['label'].names
    
    with tf.device('/job:localhost'):
        in_memory_ds = tfds.load(TFDS_NAME, batch_size=-1, shuffle_files=True)
        
    bert_preprocess_model = make_bert_preprocess_model([SENTENCE_FEATURE])

    train_dataset, train_data_size = get_data_pipeline(
        in_memory_ds, tfds_info, 'train', global_batch_size, bert_preprocess_model)

    validation_dataset, validation_data_size = get_data_pipeline(
        in_memory_ds, tfds_info, 'validation', global_batch_size, bert_preprocess_model)
    
    # Configure the model
    steps_per_epoch = train_data_size // global_batch_size
    num_train_steps = steps_per_epoch * FLAGS.epochs
    num_warmup_steps = num_train_steps // 10
    validation_steps = validation_data_size // global_batch_size
    
    with strategy.scope():
        classifier_model = build_classifier_model(NUM_CLASSES, FLAGS.dropout_ratio)
        optimizer = optimization.create_optimizer(
            init_lr=FLAGS.init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw')
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)
    
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    
    model_dir, tb_dir, checkpoint_dir = set_job_dirs()

    # Configure Keras callbacks
    callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=checkpoint_dir)]
    callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=tb_dir, update_freq='batch'))
    
    logging.info('Starting training ...')
    classifier_model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.epochs,
        validation_steps=validation_steps,
        callbacks=callbacks)
        
    # Save trained model
    logging.info('Training completed. Saving the trained model to: {}'.format(model_dir))
    classifier_model.save(model_dir)  
    
if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
