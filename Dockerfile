FROM nvcr.io/nvidia/tensorflow:19.04-py3
## MAINTAINER Paweł Sołtysiak <psoltysiak@nvidia.com>

ADD . /workspace/suction_affordance_prediction
WORKDIR /workspace/suction_affordance_prediction
RUN pip install -r requirements.txt
