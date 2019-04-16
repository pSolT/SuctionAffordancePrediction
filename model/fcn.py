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
        
        print(features[0])
        print(features[1])
        print(labels)
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False
       
        with tf.variable_scope("rgb_branch"):
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
                rgb_logits, rgb_end_points = resnet_v1.resnet_v1_101(features[0], self.num_classes, is_training=is_training)        

                #Exclude logits
                rgb_variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['rgb_branch/resnet_v1_101/logits'])
                
                # strip scope name
                rgb_assignment_map = { rgb_variables_to_restore[0].name.split(':')[0] : rgb_variables_to_restore[0]}
                rgb_assignment_map.update({ v.name.split(':')[0].split('/', 1)[1] : v for v in rgb_variables_to_restore[1:] })
      
                tf.train.init_from_checkpoint(self.pre_trained_model_path, rgb_assignment_map)
        
        
        with tf.variable_scope("depth_branch"):
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):     
                depth_logits, depth_end_points = resnet_v1.resnet_v1_101(features[1], self.num_classes, is_training=is_training)

                #Exclude rgb branch already existing in the graph and logits
                depth_variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['rgb_branch', 'depth_branch/resnet_v1_101/logits'])

                for v in depth_variables_to_restore:
                    print(v)
                
                depth_assignment_map = { depth_variables_to_restore[0].name.split(':')[0] : depth_variables_to_restore[0]}
                depth_assignment_map.update({ v.name.split(':')[0].split('/', 1)[1] : v for v in depth_variables_to_restore[1:] })
      
                tf.train.init_from_checkpoint(self.pre_trained_model_path, depth_assignment_map)

        rgb_branch_output = rgb_end_points['pre_fully_connected']
        depth_branch_output = depth_end_points['pre_fully_connected']
        
        print(rgb_branch_output)
        print(depth_branch_output)
        
        model = tf.concat( [rgb_branch_output, depth_branch_output], axis=2, name='branches_concat')
        print(model)
        model = tf.contrib.slim.conv2d(model, 512, [1, 1], scope='conv1_1')
        print(model)
        model = tf.contrib.slim.conv2d(model, 128, [1, 1], scope='conv2_1')
        print(model)
        logits = tf.contrib.slim.conv2d(model, 3, [1, 1], scope='logits')
        print(logits)
        print(labels)
        # Use Spatial Up Sampling Bilinear
        
        
        # USE Symmetric Padding
        
        # 
        # Convert end_points_collection into a dictionary of end_points.
        
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)  
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #total_loss = tf.add_n([loss] + reg_losses, name='total_loss')
                
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.99)
        train_op = optimizer.minimize(loss, global_step)
        
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss} , every_n_iter=10)
     
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks = [logging_hook])
