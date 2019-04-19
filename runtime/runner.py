# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import multiprocessing
import warnings
import math

import tensorflow as tf
import numpy as np

import horovod.tensorflow as hvd

from model import fcn

from utils import hooks
from utils import data_utils

from dllogger.logger import LOGGER

__all__ = [
    'Runner',
]

class Runner(object):

    def __init__(self, n_classes, height, width, model_dir, log_dir, data_dir, pre_trained_model_path, use_transpose_conv, use_xla=False,  use_tf_amp=False, seed=None ):


        if data_dir is not None and not os.path.exists(data_dir):
            raise ValueError("The `data_dir` received does not exists: %s" % data_dir)
        
        hvd.init()
        tf_seed = 2 * (seed + hvd.rank()) if seed is not None else None

        # ============================================
        # Optimsation Flags - Do not remove
        # ============================================

        os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'


        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(hvd.size())

        if use_tf_amp:
            if hvd.rank() == 0:
                LOGGER.log("TF AMP is activated - Experimental Feature")
            os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

        if use_tf_amp:
            if hvd.rank() == 0:
                LOGGER.log("Using transpose convolutions-based approach")
            
        model_hparams = tf.contrib.training.HParams(
            width=height,
            height=width,
            n_classes=n_classes,
            seed=tf_seed,
            use_transpose_conv=use_transpose_conv,
            label_output_scale = 1 if use_transpose_conv else 8,
            upscale_ratio = 1 if use_transpose_conv else 4
        )

        run_config_performance = tf.contrib.training.HParams(
            num_preprocessing_threads=32,
            use_tf_amp=use_tf_amp,
            use_xla=use_xla
        )

        run_config_additional = tf.contrib.training.HParams(
            model_dir=model_dir if hvd.rank() == 0 else None,
            log_dir=log_dir if hvd.rank() == 0 else None,
            data_dir=data_dir,
            pre_trained_model_path=pre_trained_model_path,
            num_preprocessing_threads=32,
        )

        self.run_hparams = Runner._build_hparams(model_hparams, run_config_additional, run_config_performance)


        self._model = fcn.FCN(
            pre_trained_model_path=run_config_additional.pre_trained_model_path,
            num_classes=model_hparams.n_classes,
            use_transpose_conv=model_hparams.use_transpose_conv,
            upscale_ratio=model_hparams.upscale_ratio
        )

        if self.run_hparams.seed is not None:
            if hvd.rank() == 0:
                LOGGER.log("Deterministic Run - Seed: %d\n" % self.run_hparams.seed)
            tf.set_random_seed(self.run_hparams.seed)
            

    @staticmethod
    def _build_hparams(*args):

        hparams = tf.contrib.training.HParams()

        for _hparams in args:
            if not isinstance(_hparams, tf.contrib.training.HParams):
                raise ValueError("Non valid HParams argument object detected:", _hparams)

            for key, val in _hparams.values().items():
                try:
                    hparams.add_hparam(name=key, value=val)

                except ValueError:
                    warnings.warn(
                        "the parameter `{}` already exists - existing value: {} and duplicated value: {}".format(
                            key, hparams.get(key), val
                        )
                    )

        return hparams
    
        
    @staticmethod
    def _get_session_config(mode, use_xla):

        if mode not in ["train", 'validation', 'benchmark']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark')" % mode)

        config = tf.ConfigProto()

        config.allow_soft_placement = True
        config.log_device_placement = False

        config.gpu_options.allow_growth = True

        config.gpu_options.visible_device_list = str(hvd.local_rank())

        if use_xla: 
            LOGGER.log("XLA is activated - Experimental Feature")
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            
            
        config.gpu_options.force_gpu_compatible = True  # Force pinned memory

        if mode == 'train':
            config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
            config.inter_op_parallelism_threads = max(2, (multiprocessing.cpu_count() // hvd.size()) - 2)

        return config
    
    
    @staticmethod
    def _get_run_config(mode, model_dir, use_xla, seed=None):

        if mode not in ["train", 'validation', 'benchmark']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark')" % mode)

        if seed is not None:
            tf_random_seed = 2 * (seed + hvd.rank())
        else:
            tf_random_seed = None

        config = tf.estimator.RunConfig(
            model_dir=model_dir,
            tf_random_seed=tf_random_seed,
            save_summary_steps=100 if mode in ['train', 'validation'] else 1e9,  # disabled in benchmark mode
            save_checkpoints_steps=None,
            save_checkpoints_secs=None,
            session_config=Runner._get_session_config(mode=mode, use_xla=use_xla),
            keep_checkpoint_max=5,
            keep_checkpoint_every_n_hours=1e6,  # disabled
            log_step_count_steps=1e9,
            train_distribute=None,
            device_fn=None,
            protocol=None,
            eval_distribute=None,
            experimental_distribute=None
        )

        if mode == 'train':
            config = config.replace(
                    save_checkpoints_steps=1000 if hvd.rank() == 0 else None, keep_checkpoint_every_n_hours=3
            )

        return config

    def _get_estimator(self, mode, run_params, use_xla):

        if mode not in ["train", 'validation', 'benchmark']:
            raise ValueError("Unknown mode received: %s (allowed: 'train', 'validation', 'benchmark')" % mode)

        run_config = Runner._get_run_config(
            mode=mode,
            model_dir=self.run_hparams.model_dir,
            use_xla=use_xla,
            seed=self.run_hparams.seed
        )

        return tf.estimator.Estimator(
            model_fn=self._model,
            model_dir=self.run_hparams.model_dir,
            config=run_config,
            params=run_params
        )

    def train(
        self,
        iter_unit,
        num_iter,
        batch_size,
        warmup_steps=50,
        weight_decay=1e-4,
        learning_rate_init=0.1,
        momentum=0.9,
        log_every_n_steps=1,
        loss_scale=256,
        use_auto_loss_scaling=False,
        is_benchmark=False
    ):

        if iter_unit not in ["epoch", "batch"]:
            raise ValueError('`iter_unit` value is unknown: %s (allowed: ["epoch", "batch"])' % iter_unit)

        if self.run_hparams.data_dir is None and not is_benchmark:
            raise ValueError('`data_dir` must be specified for training!')

        if self.run_hparams.use_tf_amp:
            if use_auto_loss_scaling:
                LOGGER.log("TF Loss Auto Scaling is activated - Experimental Feature")
                os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1"

            else:
                os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "0"

        num_gpus = hvd.size()
        global_batch_size = batch_size * num_gpus
        num_samples = data_utils.get_num_images(self.run_hparams.data_dir, mode="train")
        
        if iter_unit == 'epoch':
            num_steps = (num_samples // global_batch_size) * num_iter
            num_epochs = num_iter
            num_decay_steps = num_steps
            steps_per_epoch = num_steps / num_epochs

        else:
            num_steps = num_iter
            num_epochs = math.ceil(num_steps / (num_samples // global_batch_size))
            num_decay_steps = 90 * num_samples // global_batch_size
            steps_per_epoch = num_steps
        
        training_hooks = []
      
        if hvd.rank() == 0:
            LOGGER.log('Starting Model Training...')
            LOGGER.log("Training Epochs", num_epochs)
            LOGGER.log("Total Steps", num_steps)
            LOGGER.log("Steps per Epoch", steps_per_epoch)
            LOGGER.log("Decay Steps", num_decay_steps)
            LOGGER.log("Weight Decay Factor", weight_decay)
            LOGGER.log("Init Learning Rate", learning_rate_init)
            LOGGER.log("Momentum", momentum)
            LOGGER.log("Num GPUs", num_gpus)
            LOGGER.log("Per-GPU Batch Size", batch_size)

            training_logging_hook = hooks.TrainingLoggingHook(
                global_batch_size=global_batch_size,
                log_every=1
            )

            training_hooks.append(training_logging_hook)

            
        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
        training_hooks.append(bcast_hook)

      
        estimator_params = {
            'batch_size': batch_size,
            'num_gpus': num_gpus,
            'momentum': momentum,
            'learning_rate_init': learning_rate_init,
            'log_dir' : self.run_hparams.log_dir
        }

        image_classifier = self._get_estimator(
            mode='train',
            run_params=estimator_params,
            use_xla=self.run_hparams.use_xla
        )

        def training_data_fn():
            return data_utils.get_input_fn(
                data_dir=self.run_hparams.data_dir,
                mode='train',
                batch_size=batch_size,
                label_output_scale=self.run_hparams.label_output_scale
            )

        try:
            image_classifier.train(
                input_fn=training_data_fn,
                steps=num_steps,
                hooks=training_hooks,
            )
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            
        if hvd.rank() == 0:
            LOGGER.log('Ending Model Training ...')

    def evaluate(
        self,
        iter_unit,
        num_iter,
        batch_size,
        warmup_steps=50,
        log_every_n_steps=1,
        is_benchmark=False
    ):

        if iter_unit not in ["epoch", "batch"]:
            raise ValueError('`iter_unit` value is unknown: %s (allowed: ["epoch", "batch"])' % iter_unit)

        estimator_params = {
            'log_dir' : self.run_hparams.log_dir
        }
            
        image_classifier = self._get_estimator(
            mode='validation',
            run_params=estimator_params,
            use_xla=self.run_hparams.use_xla,
        )
        
        num_images = data_utils.get_num_images(self.run_hparams.data_dir, mode="test")
        
        if iter_unit == 'epoch':
            num_steps = (num_samples // global_batch_size) * num_iter
            num_epochs = num_iter
            num_decay_steps = num_steps

        else:
            num_steps = num_iter
            num_epochs = math.ceil(num_steps / (num_images // batch_size))
            num_decay_steps = 90 * num_images // batch_size
        
        eval_hooks = []
        
        if hvd.rank() == 0:
            if is_benchmark:
                
                benchmark_logging_hook = hooks.BenchmarkLoggingHook(
                    log_file_path=os.path.join(self.run_hparams.log_dir, "eval_benchmark.json"),
                    global_batch_size=batch_size,
                    log_every=log_every_n_steps,
                    warmup_steps=warmup_steps
                )
                eval_hooks.append(benchmark_logging_hook)

            LOGGER.log('Starting Model Evaluation...')
            LOGGER.log("Evaluation Epochs", num_epochs)
            LOGGER.log("Evaluation Steps", num_steps)
            LOGGER.log("Decay Steps", num_decay_steps)
            LOGGER.log("Global Batch Size", batch_size)

            
        def evaluation_data_fn():      
            return data_utils.get_input_fn(
                data_dir=self.run_hparams.data_dir,
                mode='test',
                batch_size=batch_size,
                label_output_scale=self.run_hparams.label_output_scale
            )
     
        try:
            eval_results = image_classifier.evaluate(
                input_fn=evaluation_data_fn,
                steps=num_steps,
                hooks=eval_hooks,
            )
            
            LOGGER.log('Top-1 Accuracy: %.3f' % float(eval_results['top1_accuracy'] * 100))
            LOGGER.log('Top-5 Accuracy: %.3f' % float(eval_results['top5_accuracy'] * 100))
            
        except KeyboardInterrupt:
            print("Keyboard interrupt")

        LOGGER.log('Ending Model Evaluation ...')
