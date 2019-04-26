# SuctionAffordancePrediction

# Setup

The following section list the requirements that you need to meet in order to use the Suction Affordance Prediction model.

## Requirements
This repository contains Dockerfile which extends the Tensorflow NGC container and encapsulates all dependencies.  Aside from these dependencies, ensure you have the following software:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorFlow 19.04-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)

# Quick start guide

## 1. Download pre-trained model checkpoint and dataset
```bash
bash prepare.sh
```

Pre-trained model checkpoint and dataset will be downloaded to directories `pre_trained_model` and `data` respectively.


## 2. Run the content of this directory inside TensorFlow 19.04-py3 NGC container

```bash
nvidia-docker run -it --rm     --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  -v `pwd`:/work   nvcr.io/nvidia/tensorflow:19.04-py3
```

## 3.Run training 

```bash
python main.py --mode train_and_evaluate --data_dir ./dataset/data/ --pre_trained_model_path ./pre_trained_model/resnet_v1_101.ckpt --num_iter 10--iter_unit epoch --results_dir /results --batch_size 8
```
To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. To summarize, the most important arguments are as follows:

```
  -h, --help            show this help message and exit
  --mode {train,train_and_evaluate,evaluate,training_benchmark,inference_benchmark}
                        The execution mode of the script.
  --data_dir DATA_DIR   Path to dataset.
  --batch_size BATCH_SIZE
                        Size of each minibatch per GPU.
  --num_iter NUM_ITER   Number of iterations to run.
  --iter_unit {epoch,batch}
                        Unit of iterations.
  --warmup_steps WARMUP_STEPS
                        Number of steps considered as warmup and not taken
                        into account for performance measurements.
  --eval_every EVAL_EVERY
                        Eval every n iterations
  --results_dir RESULTS_DIR
                        Directory in which to write training logs, summaries
                        and checkpoints.
  --pre_trained_model_path PRE_TRAINED_MODEL_PATH
                        Path to pre-trained ResNet-v1-101
  --display_every DISPLAY_EVERY
                        How often (in batches) to print out running
                        information.
  --lr_init LR_INIT     Initial value for the learning rate.
  --weight_decay WEIGHT_DECAY
                        Weight Decay scale factor.
  --momentum MOMENTUM   SGD momentum value for the Momentum optimizer.
  --use_transpose_conv  Use transpose convolutions based approach
  --use_tf_amp          Enable Automatic Mixed Precision to speedup FP32
                        computation using tensor cores.
  --use_xla             Enable XLA (Accelerated Linear Algebra) computation
                        for improved performance.
  --seed SEED           Random seed.
```