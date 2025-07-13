import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import argparse
import time
import sys

sys.path.append('/gpfs/scratch/nct_325/cifar-utils')
from cifar import load_cifar

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2048)

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs

train_ds, test_ds = load_cifar(batch_size)

model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None,
            input_shape=(128, 128, 3), classes=10)

opt = tf.keras.optimizers.SGD(0.01)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(train_ds, epochs=epochs, verbose=2)
