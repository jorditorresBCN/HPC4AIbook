import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import sys
from six.moves import cPickle
import os

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_cifar(batch_size, path='/gpfs/scratch/nct_325/cifar-utils/cifar-10-batches-py'):
    train_images = np.empty((50000, 3, 32, 32), dtype='uint8')
    train_labels = np.empty((50000,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (train_images[(i - 1) * 10000: i * 10000, :, :, :],
            train_labels[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    test_images, test_labels = load_batch(fpath)
    ## Load as numpy
    # train_labels = np.reshape(train_labels, (len(train_labels), 1))
    # test_labels = np.reshape(test_labels, (len(test_labels), 1))

    # if K.image_data_format() == 'channels_last':
    #     train_images = train_images.transpose(0, 2, 3, 1)
    #     test_images = test_images.transpose(0, 2, 3, 1)

    # # Normalize data.
    # train_images = train_images.astype('float32') / 255
    # test_images = test_images.astype('float32') / 255

    # # Convert class vectors to binary class matrices.
    # train_labels = to_categorical(train_labels, 10)
    # test_labels = to_categorical(test_labels, 10)

    ## Load as tf.data.Dataset
    def preprocess_fn(image, label):

        label = tf.cast(label, tf.int32)

        if K.image_data_format() == 'channels_last':
            image = tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0])
            
        #resize image
        image = tf.image.resize(image, [128,128])

        # Normalize data.
        image = tf.math.divide(tf.cast(image, tf.float32), 255)

        return image, label
        
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_ds = train_ds.map(preprocess_fn, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().repeat(1)
    test_ds = test_ds.map(preprocess_fn, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds
