#!/bin/bash
set -e

mkdir -p "pre_trained_model"
echo "Downloading pre-trained ResNet-v1-101 model"
curl -O http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar xvzf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt pre_trained_model
rm resnet_v1_101_2016_08_28.tar.gz

mkdir -p "dataset"
echo "Downloading dataset"
wget http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip
unzip suction-based-grasping-dataset.zip
rm suction-based-grasping-dataset.zip


echo "Pulling docker image"
docker pull nvcr.io/nvidia/tensorflow:19.04-py3