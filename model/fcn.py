from __future__ import print_function

import tensorflow as tf

import os

import horovod.tensorflow as hvd

from utils.optimizers import FixedLossScalerOptimizer

from model import resnet_v1

from dllogger.logger import LOGGER

class FCN(object):
    def __init__(self, pre_trained_model_path, num_classes):
        self.pre_trained_model_path = pre_trained_model_path
        self.num_classes = num_classes
        
    def __call__(self, features, labels, mode, params):

        rgb_inputs = features[0] 
        depth_inputs = features[1] 
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False
       
        with tf.variable_scope("rgb_branch"):
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
                rgb_logits, rgb_end_points = resnet_v1.resnet_v1_101(rgb_inputs, self.num_classes, is_training=is_training)        

                #Exclude logits
                rgb_variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['rgb_branch/resnet_v1_101/logits'])
                
                #Strip scope name
                rgb_assignment_map = { rgb_variables_to_restore[0].name.split(':')[0] : rgb_variables_to_restore[0]}
                rgb_assignment_map.update({ v.name.split(':')[0].split('/', 1)[1] : v for v in rgb_variables_to_restore[1:] })
      
                tf.train.init_from_checkpoint(self.pre_trained_model_path, rgb_assignment_map)
        
        
        with tf.variable_scope("depth_branch"):
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):     
                depth_logits, depth_end_points = resnet_v1.resnet_v1_101(depth_inputs, self.num_classes, is_training=is_training)

                #Exclude rgb branch already existing in the graph and logits
                depth_variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['rgb_branch', 'depth_branch/resnet_v1_101/logits'])
                
                depth_assignment_map = { depth_variables_to_restore[0].name.split(':')[0] : depth_variables_to_restore[0]}
                depth_assignment_map.update({ v.name.split(':')[0].split('/', 1)[1] : v for v in depth_variables_to_restore[1:] })
      
                tf.train.init_from_checkpoint(self.pre_trained_model_path, depth_assignment_map)

        rgb_branch_output = rgb_end_points['pre_fully_connected']
        depth_branch_output = depth_end_points['pre_fully_connected']
        
        model = tf.concat( [rgb_branch_output, depth_branch_output], axis=3, name='branches_concat')
        
        model = tf.contrib.slim.conv2d_transpose(model, 512, (3, 3), stride=4, padding='SAME')
        model = tf.contrib.slim.conv2d_transpose(model, 128, (4, 4), stride=4, padding='SAME')
        model = tf.contrib.slim.conv2d_transpose(model, 3, (4, 4), stride=2, padding='SAME')
        
        logits = tf.nn.relu(model)

        #Flatten logits and labels to calculate loss
        logits_flat = tf.reshape(logits, [-1])
        labels_flat = tf.reshape(labels, [-1])
        
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_flat, labels=labels_flat)
                
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.00001, momentum=0.99)
        train_op = optimizer.minimize(loss, global_step)
        
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss} , every_n_iter=1)
     
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks = [logging_hook])
