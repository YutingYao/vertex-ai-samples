
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-5

RUN pip install tf-models-official==2.5.0 tensorflow-text==2.5.0

WORKDIR /

# Copies the trainer code to the docker image.
COPY nlp-trainer /trainer

ENTRYPOINT ["python"]
CMD ["-c", "print('Hello')"]
