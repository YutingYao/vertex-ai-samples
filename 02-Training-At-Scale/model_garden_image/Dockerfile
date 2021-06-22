FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-5

RUN apt remove -y google-fast-socket \
&&  echo "deb https://packages.cloud.google.com/apt google-fast-socket main" | tee /etc/apt/sources.list.d/google-fast-socket.list \
&&  curl -s -L https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
&&  apt update && apt install -y google-reduction-server

RUN pip install tf-models-official==2.5.0 tensorflow-text==2.5.0

WORKDIR /

COPY trainer /trainer
COPY dataprep /dataprep

ENTRYPOINT ["python"]
CMD ["-c", "print('TF Model Garden')"]
