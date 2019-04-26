import time

import numpy as np
import tensorflow as tf

__all__ = ['TrainingLoggingHook']

import dllogger.logger as dllg
from dllogger.logger import LOGGER

class TrainingLoggingHook(tf.train.SessionRunHook):

    def __init__( self, global_batch_size, log_every=10):
        
        LOGGER.set_model_name('SuctionAffordancePredictor')

        LOGGER.set_backends(
            [
                dllg.StdOutBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=log_every)
            ]
        )
        
        # Set-up the train_iter scope metrics
        LOGGER.register_metric("iteration", metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("imgs_per_sec", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)
        LOGGER.register_metric("cross_entropy", meter=dllg.AverageMeter(), metric_scope=dllg.TRAIN_ITER_SCOPE)

        self.global_batch_size = global_batch_size
        self.current_step = 0
        self.current_epoch = 0

    def before_run(self, run_context):
        LOGGER.iteration_start()
        '''
        run_args = tf.train.SessionRunArgs(
            fetches=[
                tf.train.get_global_step(), 
                'cross_entropy_loss_ref:0', 
                'logits_flat_ref:0',
                'labels_flat_ref:0',
                'rgb_inputs:0',
                'depth_inputs:0',
                'labels_flat_ref:0',
                'logits_flat_ref:0',
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

        #global_step, cross_entropy, logits_flat, labels_flat, rgb_inputs, depth_inputs, labels, logits = run_values.results
        global_step, cross_entropy = run_values.results
        batch_time = time.time() - self.t0
        ips = self.global_batch_size / batch_time

        LOGGER.log("iteration", int(self.current_step))
        LOGGER.log("imgs_per_sec", float(ips))
        LOGGER.log("cross_entropy", float(cross_entropy))
        LOGGER.iteration_stop()
        self.current_step += 1

        #print("RGB Inputs: Mean", np.mean(rgb_inputs), " std: ", np.std(rgb_inputs), "min", np.min(rgb_inputs), "max", np.max(rgb_inputs))
        #print("Depth Inputs: Mean", np.mean(depth_inputs), " std: ", np.std(depth_inputs), "min", np.min(depth_inputs), "max", np.max(depth_inputs))
        #print("Logits: Mean", np.mean(logits), " std: ", np.std(logits), "min", np.min(logits), "max", np.max(logits))        
        #print("Labels: Mean", np.mean(labels), " std: ", np.std(labels), "min", np.min(labels), "max", np.max(labels))
        #for i in range(len(logits_flat)):
            #print(logits_flat[i], labels_flat[i])

    def end(self, session):
        LOGGER.finish()
