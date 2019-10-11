FROM pytorch/pytorch

RUN apt-get -y update && apt-get -y install ffmpeg && apt-get -y install mpich && apt-get -y install libsm6 libxext6 libxrender-dev
# RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv

ENV CODE_DIR /root/code

COPY . $CODE_DIR/AGNES
WORKDIR $CODE_DIR/AGNES

RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install opencv-python && \
    pip install -r requirements.txt && \
    pip install .


CMD /bin/bash
