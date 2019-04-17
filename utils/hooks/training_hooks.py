import time

import numpy as np
import tensorflow as tf

__all__ = ['TrainingLoggingHook']


class TrainingLoggingHook(tf.train.SessionRunHook):

    def __init__(
        self, global_batch_size, log_every=10, warmup_steps=20
    ):
        self.global_batch_size = global_batch_size
        self.current_step = 0
        self.current_epoch = 0

    # Determines if its the last step of the epoch
    def _last_step_of_epoch(self):
        return (self.global_batch_size * (self.current_step + 1) > (self.current_epoch + 1) * self.num_samples)

    def before_run(self, run_context):
        '''
        run_args = tf.train.SessionRunArgs(
            fetches=[
                tf.train.get_global_step(), 
                'cross_entropy_loss_ref:0', 
                'logits_flat_ref:0',
                'labels_flat_ref:0',
                'rgb_inputs:0',
                'depth_inputs:0',
                'labels:0'
            ]
        )
        '''
        run_args = tf.train.SessionRunArgs(
            fetches=[
                tf.train.get_global_step(), 
                'cross_entropy_loss_ref:0',
            ]
        )
        self.t0 = time.time()
        return run_args

    def after_run(self, run_context, run_values):
        
        #global_step, cross_entropy, logits_flat, labels_flat, rgb_inputs, depth_inputs, labels = run_values.results
        global_step, cross_entropy = run_values.results
        batch_time = time.time() - self.t0
        ips = self.global_batch_size / batch_time
        print(global_step, batch_time, cross_entropy)
        
        #print("RGB Inputs: Mean", np.mean(rgb_inputs), " std: ", np.std(rgb_inputs), "min", np.min(rgb_inputs), "max", np.max(rgb_inputs))
        #print("Depth Inputs: Mean", np.mean(depth_inputs), " std: ", np.std(depth_inputs), "min", np.min(depth_inputs), "max", np.max(depth_inputs))
        #print("Labels: Mean", np.mean(labels), " std: ", np.std(labels), "min", np.min(labels), "max", np.max(labels))
        #for i in range(len(logits_flat)):
            #print(logits_flat[i], labels_flat[i])

