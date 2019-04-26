from __future__ import print_function

import tensorflow as tf

import os

import horovod.tensorflow as hvd

from model import resnet_v1

class FCN(object):
    
    def __init__(self, pre_trained_model_path, num_classes, upscale_ratio=4, use_transpose_conv=True):
        self.pre_trained_model_path = pre_trained_model_path
        self.num_classes = num_classes
        self.upscale_ratio = upscale_ratio
        self.use_transpose_conv = use_transpose_conv
        
    def __call__(self, features, labels, mode, params):

        rgb_inputs = features[0] 
        depth_inputs = features[1]
            
        # Needed to easily fetch values for debugging
        rgb_inputs = tf.identity(rgb_inputs, name='rgb_inputs')
        depth_inputs = tf.identity(depth_inputs, name='depth_inputs')
        labels = tf.identity(labels, name='labels')
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
        else:
            is_training = False
       
        with tf.device("/gpu:0"):
            # Create RGB and Depth branches and load the weights from checkpoint
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
            rgb_logits = tf.contrib.slim.conv2d(rgb_branch_output, 3, [1, 1], padding='SAME')
            
            depth_branch_output = depth_end_points['pre_fully_connected']        
            depth_logits = tf.contrib.slim.conv2d(depth_branch_output, 3, [1, 1], padding='SAME')

            model = tf.concat( [rgb_branch_output, depth_branch_output], axis=3, name='branches_concat')

            # Use standard convolutiomns and upscale with bilinear resize
            if self.upscale_ratio and not self.use_transpose_conv:
                model = tf.contrib.slim.conv2d(model, 512, [1, 1], padding='SAME')
                model = tf.contrib.slim.conv2d(model, 128, [1, 1], padding='SAME')
                model = tf.contrib.slim.conv2d(model, 3, [1, 1], padding='SAME')

                height = tf.shape(model)[1]
                width = tf.shape(model)[2]

                logits = tf.image.resize_bilinear(model, [height * self.upscale_ratio, width * self.upscale_ratio])

            # Use transpose convolutions to upscale to full label size
            else:
                model = tf.contrib.slim.conv2d_transpose(model, 512, (3, 3), stride=4, padding='SAME', activation_fn=None)
                model = tf.contrib.slim.conv2d_transpose(model, 128, (4, 4), stride=4, padding='SAME', activation_fn=None)
                model = tf.contrib.slim.conv2d_transpose(model, 3, (4, 4), stride=2, padding='SAME', activation_fn=None)

                logits = tf.nn.relu(model)

    
            tf.summary.image("rgb_input", features[0])
            tf.summary.image("depth_input", features[1])
            tf.summary.image("labels", features[2])
            tf.summary.image("logits", logits)
            summaries = tf.summary.merge_all()

            # Flatten logits and labels to calculate loss
            logits_flat = tf.reshape(logits, [-1, 3])  
            labels_flat = tf.reshape(labels, [-1])

            logits_flat = tf.identity(logits_flat, name='logits_flat_ref')
            
            # Transform labels to one-hot encoding
            labels_flat = tf.one_hot(labels_flat, depth=3, dtype=tf.float32)
            labels_flat = tf.identity(labels_flat, name='labels_flat_ref')       
                
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_flat, labels=labels_flat))
            loss = tf.identity(loss, name='cross_entropy_loss_ref')

            clipped_logits = tf.clip_by_value(logits_flat, 0.0, 0.9999, name='clipped_logits')
            
            tp, update_tp = tf.metrics.true_positives(
                labels=labels_flat,
                predictions=clipped_logits
            )
            
            fp, update_fp = tf.metrics.false_positives(
                labels=labels_flat,
                predictions=clipped_logits
            )
            
            precision = tp / (tp + fp)
                
            tf.summary.scalar('precision', precision) 

            if mode == tf.estimator.ModeKeys.TRAIN:
            
                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
                optimizer = hvd.DistributedOptimizer(optimizer)
            
                train_op = tf.contrib.training.create_train_op(
                    total_loss = loss,
                    optimizer = optimizer,
                    global_step = tf.train.get_or_create_global_step())

                logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "precision": precision} , every_n_iter=10)
                
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    training_hooks = [logging_hook])
    
            elif mode == tf.estimator.ModeKeys.EVAL:
                                
                eval_metrics = {}
                eval_metrics["true_positives"] = tp, update_tp
                eval_metrics["false_positives"] = fp, update_fp
                
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=clipped_logits,
                    loss=loss,
                    eval_metric_ops=eval_metrics
                )