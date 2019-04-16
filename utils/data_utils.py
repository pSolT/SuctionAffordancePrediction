import sys
import os 
import math 

import tensorflow as tf
import horovod.tensorflow as hvd


__all__ = ["get_input_fn", "get_num_images"]

#local mean = {0.485,0.456,0.406}
#local std = {0.229,0.224,0.225}
#Each calculated * 256
_R_MEAN = 124.16
_G_MEAN = 116.73
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

_R_STD = 58.62
_G_STD = 57.344
_B_STD = 57.6
_CHANNEL_STDS = [_R_STD, _G_STD, _B_STD]

_NUM_CHANNELS = 3

def get_splits(data_dir):
    test_split_path = os.path.join(data_dir, 'test-split.txt')
    train_slit_path = os.path.join(data_dir, 'train-split.txt')
    
    with open(test_split_path) as file:
        test_split = (line.rstrip('\n,') for line in file)
        test_split = [ r for r in test_split ]
        
    with open(train_slit_path) as file:
        train_split = (line.rstrip('\n,') for line in file)
        train_split = [ r for r in train_split ]
        
    return (train_split, test_split)


def get_num_images(data_dir, mode='train'):
    if mode == 'train':
        train_slit_path = os.path.join(data_dir, 'train-split.txt')
        with open(train_slit_path) as file:
            train_split = (line.rstrip('\n,') for line in file)
            train_split = [ r for r in train_split ]
        return len(train_split)
    
    elif mode == 'test':
        test_split_path = os.path.join(data_dir, 'test-split.txt')
        with open(test_split_path) as file:
            test_split = (line.rstrip('\n,') for line in file)
            test_split = [ r for r in test_split ]
        return len(test_split)
            
    else:
        raise ValueError("Mode must be eiher 'train' or 'test'") 

def get_dataset_files(data_dir, dataset_name, split):
    dataset_dir = os.path.join(data_dir, dataset_name)
    dataset_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.splitext(os.path.basename(f))[0] in split]
    
    return dataset_files

def read_image(path):
    file =  tf.read_file(path)
    return file

def preprocess_color_image(image):
    image = tf.image.decode_png(image, channels=3, dtype=tf.dtypes.uint8)
    image = tf.cast(image, tf.dtypes.float32)
    means_per_channel = tf.reshape(_CHANNEL_MEANS, [1, 1, _NUM_CHANNELS])
    means_per_channel = tf.cast(means_per_channel, dtype=image.dtype)

    stds_per_channel = tf.reshape(_CHANNEL_STDS, [1, 1, _NUM_CHANNELS])
    stds_per_channel = tf.cast(means_per_channel, dtype=image.dtype)
    
    image = tf.subtract(image, means_per_channel)
    image = tf.divide(image, stds_per_channel)

    return image


def preprocess_depth_image(image):
    image = tf.image.decode_png(image, channels=1, dtype=tf.dtypes.uint8)
    image = tf.cast(image, tf.dtypes.float32)
    image = tf.multiply(image, 65536/10000)
    image = tf.clip_by_value(image, 0.0, 1.2)
    

    depth_image = tf.concat([image, image, image], 2)

    means_per_channel = tf.reshape(_CHANNEL_MEANS, [1, 1, _NUM_CHANNELS])
    stds_per_channel = tf.reshape(_CHANNEL_STDS, [1, 1, _NUM_CHANNELS])
    
    depth_image = tf.subtract(depth_image, means_per_channel)
    depth_image = tf.divide(depth_image, stds_per_channel)
    return depth_image

    
def preprocess_label_image(image):
    image = tf.image.decode_png(image, channels=3, dtype=tf.dtypes.uint8)
    image = tf.cast(image, tf.dtypes.float32)
    image = tf.multiply(image, 2.0)
    image = tf.round(image)
    image = tf.add(image, 1.0)
    
    return image
    

def get_input_fn(data_dir, batch_size, mode='train'):
    shuffle_buffer_size = 10
    
    train_split, test_split = get_splits(data_dir)
    training = False
    
    if mode == 'train':
        split = train_split
        training = True
        
    elif mode == 'test':
        split = test_split
        
    else:
        raise ValueError("Mode must be eiher 'train' or 'test'") 
       
    color_input_files = get_dataset_files(data_dir, 'color-input', split)
    depth_input_files = get_dataset_files(data_dir, 'depth-input', split)
    label_files = get_dataset_files(data_dir, 'label', split)

    dataset =  tf.data.Dataset.from_tensor_slices((color_input_files, depth_input_files, label_files))
    def load_and_preprocess_from_paths(color_input_file_path, depth_input_file_path, label_file_path):
        return (preprocess_color_image(read_image(color_input_file_path)), preprocess_depth_image(read_image(color_input_file_path))),  preprocess_label_image(read_image(color_input_file_path))
            
    dataset = dataset.map(load_and_preprocess_from_paths, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if training:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=shuffle_buffer_size))
    else:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    
    return dataset
    
    

