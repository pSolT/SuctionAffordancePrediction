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

import argparse


def parse_cmdline():

    p = argparse.ArgumentParser(description="SuctionAffordancePrediction")

    p.add_argument(
        '--mode',
        choices=['train', 'train_and_evaluate', 'evaluate', 'training_benchmark', 'inference_benchmark'],
        type=str,
        default='train_and_evaluate',
        required=False,
        help="""The execution mode of the script."""
    )

    p.add_argument(
        '--data_dir',
        required=False,
        default=None,
        type=str,
        help="Path to dataset in TFRecord format. Files should be named 'train-*' and 'validation-*'."
    )
    
    p.add_argument(
        '--image_height', 
        type=int, 
        default=480,
        required=False, 
        help="""Height of an dataset image"""
    )
    
    p.add_argument(
        '--image_width', 
        type=int, 
        default=640,
        required=False,  
        help="""Width of an dataset image"""
    )

    p.add_argument(
        '--batch_size', 
        type=int, 
        default=2,
        required=False, 
        help="""Size of each minibatch per GPU."""
    )

    p.add_argument(
        '--num_iter',
        type=int, 
        required=True, 
        help="""Number of iterations to run."""
    )

    p.add_argument(
        '--iter_unit',
        choices=['epoch', 'batch'],
        type=str,
        required=True,
        help="""Unit of iterations."""
    )

    p.add_argument(
        '--warmup_steps',
        default=50,
        type=int,
        required=False,
        help="""Number of steps considered as warmup and not taken into account for performance measurements."""
    )
    

    p.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help="""Directory in which to write training logs, summaries and checkpoints."""
    )

    p.add_argument(
        '--display_every', 
        default=10,
        type=int, 
        required=False, 
        help="""How often (in batches) to print out running information."""
    )

    p.add_argument(
        '--lr_init',
        default=0.001,
        type=float,
        required=False,
        help="""Initial value for the learning rate."""
    )

    p.add_argument(
        '--weight_decay', 
        default=1e-4,
        type=float, 
        required=False, 
        help="""Weight Decay scale factor."""
    )

    p.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        required=False,
        help="""SGD momentum value for the Momentum optimizer."""
    )
    
    p.add_argument(
        "--use_tf_amp",
        action='store_true',
        required=False,
        help="Enable Automatic Mixed Precision to speedup FP32 computation using tensor cores."
    )
    
    p.add_argument(
        "--use_auto_loss_scaling",
        action='store_true',
        required=False,
        help="Use AutoLossScaling in TF AMP mode."
    )
    
    p.add_argument(
        "--use_xla",
        action='store_true',
        required=False,
        help="Enable XLA (Accelerated Linear Algebra) computation for improved performance."
    )
    
    p.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        help="""Random seed."""
    )

    FLAGS, unknown_args = p.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS
