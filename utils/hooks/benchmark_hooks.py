import time

import numpy as np
import tensorflow as tf

__all__ = ['BenchmarkLoggingHook']


class BenchmarkLoggingHook(tf.train.SessionRunHook):

    def __init__(self, log_file_path, global_batch_size, log_every=10, warmup_steps=20):

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
            print("iteration", int(self.current_step), "total_ips", float(ips))
        self.current_step += 1