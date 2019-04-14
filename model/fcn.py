from __future__ import print_function

import tensorflow as tf

import os

import horovod.tensorflow as hvd

from utils.learning_rate import learning_rate_scheduler
from utils.optimizers import FixedLossScalerOptimizer

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

from dllogger.logger import LOGGER

class FCN(object):
    def __init__(self, pre_trained_model_path, num_classes):
        self.pre_trained_model_path = pre_trained_model_path
        self.num_classes = num_classes
        
    def __call__(self, features, labels, mode, params):
        
        print(features[0])
        print(features[1])
        print(labels)
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False
            
        with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, end_points = resnet_v1.resnet_v1_101(features[0], self.num_classes, is_training=is_training)
            print(logits)
            print(end_points)
            
                    
            predictions = {
              'classes': tf.argmax(input=logits, axis=1),
              'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
            }
        
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
            accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
            tf.summary.scalar('accuracy', accuracy[1])
        
            # Restore all the variables except from the last layer excluding fully connected layers and removing batch norm


            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
            scopes = { os.path.dirname(v.name) for v in variables_to_restore }
            tf.train.init_from_checkpoint(pre_trained_model_path, 
                                  {v.name.split(':')[0]: v for v in variables_to_restore})


            # Get a handle to last variable and initalize it from scratch
            fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
            fc8_init = tf.variables_initializer(fc8_variables)

            loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)  
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n([loss] + reg_losses, name='total_loss')   

            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = {'eval_accuracy': accuracy}
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            # Re-train the last layer of model
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=0.00016)
            train_op = optimizer.minimize(total_loss, global_step, var_list=fc8_variables)

            logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 'total_loss' : total_loss}, every_n_iter=10)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=total_loss,
                train_op=train_op,
                training_hooks = [logging_hook])


