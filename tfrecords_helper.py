# -*- coding: utf-8 -*-

"""
Created on Sun Jul  21 15:57:35 2019

@author: Meet
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import config
from imgaug import augmenters as iaa

seq = iaa.Sequential([
        iaa.Fliplr(p=0.5),
        iaa.ContrastNormalization((0.75, 1.5)),
        iaa.Multiply((0.8, 1.2), per_channel=True),
        iaa.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                          rotate=(-25, 25),
                          shear=(-8, 8),
                          translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)})
        ], random_order=True)
    
    
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_tfrecords_writer(tfrecords_fn, file_counter, files_per_record, splits):
    tf_counter = file_counter // files_per_record
    tfrecords_fn = tfrecords_fn[:-9] + "{:04d}".format(tf_counter) + "-" + "{:04d}".format(splits) + ".tfrecords"
    return tf.python_io.TFRecordWriter(tfrecords_fn)

def write_tfrecords(tfrecords_path, file_lines, splits=10):    
    files_per_record = len(file_lines) // splits
    
    for i in tqdm(range(len(file_lines))):
        if i % files_per_record == 0:
            writer = get_tfrecords_writer(tfrecords_path, i, files_per_record, splits)
    
        label = np.zeros(shape=[config.num_classes], dtype=np.float32)
        line = file_lines[i]
        line = line.replace('\n', '') # to remove \n last character
        
        file = line.split(' ')[0]
        with tf.gfile.FastGFile(file, 'rb') as fid:
            img_data = fid.read()

        label_id = int(line.split(' ')[1])
        label[label_id] = 1
        label = np.float64(label)
        
        feature = {'image' : _bytes_feature(img_data),
                   'label' : _bytes_feature(label.tostring())}
        
        example = tf.train.Example(features = tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

        if (i + 1) % files_per_record == 0:
            writer.close()    
    writer.close()

#write_tfrecords("./data/train.tfrecords", config.train_file_lines, splits=100)
#write_tfrecords("./data/test.tfrecords", config.test_file_lines, splits=10)

def _extract_fxn(tfrecord_file):
    features = {'image': tf.FixedLenFeature([], tf.string),
                'label' : tf.FixedLenFeature([], tf.string)}
    
    sample = tf.parse_single_example(tfrecord_file, features)
    img = tf.image.decode_image(sample['image'], channels=3)
    label = tf.decode_raw(sample['label'], tf.float64)
    return img, label
    
def augment_data(batch_img, batch_label, augment=False):
    if augment:
        batch_img = seq(images=batch_img)
    return np.float32(batch_img / 255.), np.float32(batch_label)
    
def get_batch(tfrecords_file_list, batch_size, augment=False, is_validation_set=False):
    with tf.variable_scope('dataset_helper'):        
        all_dataset = [tf.data.TFRecordDataset([tfrecords]) for tfrecords in tfrecords_file_list]
        dataset_len = 0
        for tfrecord in tfrecords_file_list:
            dataset_len += sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord))
        
        dataset = tf.data.experimental.sample_from_datasets(all_dataset)
        dataset = dataset.shuffle(buffer_size=dataset_len)
        dataset = dataset.map(_extract_fxn)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.map(
                            lambda x, y: tf.py_func(augment_data, 
                                     inp=[x, y, augment],
                                     Tout=[tf.float32, tf.float32]))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if is_validation_set:
            dataset = dataset.repeat(None)
            iterator = dataset.make_one_shot_iterator()
            img, label = iterator.get_next()
            return img, label
        else:
            dataset_init = dataset.make_initializable_iterator()
            img, label = dataset_init.get_next()
            return dataset_init, img, label

"""
img_, label_ = get_batch(config.test_tfrecord_list, 20, augment=False, is_validation_set=True)


with tf.Session() as sess:
    img_1, label_1 = sess.run([img_, label_])

from PIL import Image
Image.fromarray(np.uint8(img_1[0] * 255)).show()
"""
    