# !/usr/bin/env python
# -*- coding: utf-8 -*-

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

import os

import warnings
warnings.simplefilter("ignore")

import tensorflow as tf

import horovod.tensorflow as hvd

from runtime import Runner

from utils.cmdline_helper import parse_cmdline

_NUM_CLASSES = 3 

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)

    FLAGS = parse_cmdline()
    
    runner = Runner(
        # ========= Model HParams ========= #
        n_classes=_NUM_CLASSES,
        
        log_dir=FLAGS.results_dir,
        model_dir=FLAGS.results_dir,
        data_dir=FLAGS.data_dir,
        pre_trained_model_path=FLAGS.pre_trained_model_path,
        use_transpose_conv=FLAGS.use_transpose_conv,
        
        # ======= Optimization HParams ======== #
        use_xla=FLAGS.use_xla,
        use_tf_amp=FLAGS.use_tf_amp,
        
        seed=FLAGS.seed
    )
    
    if FLAGS.mode == "train_and_evaluate" and FLAGS.eval_every > 0:

        for i in range(int(FLAGS.num_iter / FLAGS.eval_every)):
            
            runner.train(
                iter_unit=FLAGS.iter_unit,
                num_iter=FLAGS.eval_every,
                batch_size=FLAGS.batch_size,
                warmup_steps=FLAGS.warmup_steps,
                weight_decay=FLAGS.weight_decay,
                learning_rate_init=FLAGS.lr_init,
                momentum=FLAGS.momentum,
                is_benchmark=FLAGS.mode == 'training_benchmark'
            )
                
            runner.evaluate(
                iter_unit= "epoch",
                num_iter= 1,
                warmup_steps=FLAGS.warmup_steps,
                batch_size=FLAGS.batch_size,
                is_benchmark=FLAGS.mode == 'inference_benchmark'
            )
        
    else:  

        if FLAGS.mode in ["train", "train_and_evaluate", "training_benchmark"]:
            runner.train(
                iter_unit=FLAGS.iter_unit,
                num_iter=FLAGS.num_iter,
                batch_size=FLAGS.batch_size,
                warmup_steps=FLAGS.warmup_steps,
                weight_decay=FLAGS.weight_decay,
                learning_rate_init=FLAGS.lr_init,
                momentum=FLAGS.momentum,
                is_benchmark=FLAGS.mode == 'training_benchmark',
            )

        if FLAGS.mode in ["train_and_evaluate", 'evaluate', 'inference_benchmark']:

            if hvd.rank() == 0:

                runner.evaluate(
                    iter_unit=FLAGS.iter_unit if FLAGS.mode != "train_and_evaluate" else "epoch",
                    num_iter=FLAGS.num_iter if FLAGS.mode != "train_and_evaluate" else 1,
                    warmup_steps=FLAGS.warmup_steps,
                    batch_size=FLAGS.batch_size,
                    is_benchmark=FLAGS.mode == 'inference_benchmark'
                )
