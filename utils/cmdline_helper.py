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
        help="Path to dataset."
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
        '--eval_every',
        default=0,
        type=int,
        required=False,
        help="""Eval every n iterations"""
    )

    p.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help="""Directory in which to write training logs, summaries and checkpoints."""
    )
    
    p.add_argument(
        '--pre_trained_model_path',
        type=str,
        required=True,
        help="""Path to pre-trained ResNet-v1-101"""
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
        default=10e-7,
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
        default=0.99,
        type=float,
        required=False,
        help="""SGD momentum value for the Momentum optimizer."""
    )
    
    p.add_argument(
        "--use_transpose_conv",
        action='store_true',
        required=False,
        help="Use transpose convolutions based approach"
    )
    
    p.add_argument(
        "--use_tf_amp",
        action='store_true',
        required=False,
        help="Enable Automatic Mixed Precision to speedup FP32 computation using tensor cores."
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
        required=False,
        help="""Random seed."""
    )

    FLAGS, unknown_args = p.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS
