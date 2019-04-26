import time

import numpy as np
import tensorflow as tf

import dllogger.logger as dllg
from dllogger.logger import LOGGER

__all__ = ['BenchmarkLoggingHook']


class BenchmarkLoggingHook(tf.train.SessionRunHook):

    def __init__(self, log_file_path, global_batch_size, log_every=10, warmup_steps=20):
        LOGGER.set_model_name('SuctionAffordancePredictor')

        LOGGER.set_backends(
            [
                dllg.JoCBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=log_every)
            ]
        )

        LOGGER.register_metric("iteration", metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("total_ips", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)

        self.warmup_steps = warmup_steps
        self.global_batch_size = global_batch_size
        self.current_step = 0

    def before_run(self, run_context):
        self.t0 = time.time()
        if self.current_step >= self.warmup_steps:
            LOGGER.iteration_start()

    def after_run(self, run_context, run_values):
        batch_time = time.time() - self.t0
        ips = self.global_batch_size / batch_time
        if self.current_step >= self.warmup_steps:
            LOGGER.log("iteration", int(self.current_step))
            LOGGER.log("total_ips", float(ips))
            LOGGER.iteration_stop()

        self.current_step += 1
        
    def end(self, session):
        LOGGER.finish()
