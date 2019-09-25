FROM python:3.6

RUN apt-get -y update && apt-get -y install ffmpeg && apt-get -y install mpich
# RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv

ENV CODE_DIR /root/code

COPY . $CODE_DIR/AGNES
WORKDIR $CODE_DIR/AGNES

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -r requirements.txt


CMD /bin/bash