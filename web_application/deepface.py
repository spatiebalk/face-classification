# # Implementation of DeepFace using Keras
import keras
import numpy as np
from os import path
from os import listdir
from os.path import isfile, join
from PIL import Image 
from keras.models import Model
import tensorflow as tf
import csv
import pandas as pd
from tqdm import tqdm
import keras.initializers

IMAGE_SIZE = (152, 152) # set by the model 
CHANNELS = 3 # RGB image
NUM_CLASSES = 8631 # classification layer will be removed 
LEARN_RATE = 0.01
MOMENTUM = 0.9

DOWNLOAD_PATH = 'https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'
MD5_HASH = '0b21fb70cd6901c96c19ac14c9ea8b89'

wt_init = tf.random_normal_initializer(mean=0, stddev=0.01)
bias_init = tf.constant_initializer(value=0.5)

def create_classifying_deepface(image_size=IMAGE_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES, learn_rate=LEARN_RATE, momentum=MOMENTUM):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(*image_size, channels), name='I0'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=11, activation=tf.nn.relu, kernel_initializer=wt_init, bias_initializer=bias_init, name='C1'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same', name='M2'))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=9, activation=tf.nn.relu, kernel_initializer=wt_init, bias_initializer=bias_init, name='C3'))
    model.add(tf.keras.layers.LocallyConnected2D(filters=16, kernel_size=9, activation=tf.nn.relu, kernel_initializer=wt_init, bias_initializer=bias_init, name='L4'))
    model.add(tf.keras.layers.LocallyConnected2D(filters=16, kernel_size=7, strides=2,  activation=tf.nn.relu, kernel_initializer=wt_init, bias_initializer=bias_init, name='L5'))
    model.add(tf.keras.layers.LocallyConnected2D(filters=16, kernel_size=5, activation=tf.nn.relu, kernel_initializer=wt_init, bias_initializer=bias_init, name='L6'))
    model.add(tf.keras.layers.Flatten(name='F7'))
    model.add(tf.keras.layers.Dense(units=4096, activation=tf.nn.relu, kernel_initializer=wt_init, bias_initializer=bias_init, name='F8'))
    model.add(tf.keras.layers.Dropout(rate=0.5, name='D9'))
    model.add(tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax, kernel_initializer=wt_init, bias_initializer=bias_init, name='F10'))

    sgd_opt = tf.keras.optimizers.SGD(learning_rate=learn_rate, momentum=momentum)
    cce_loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(optimizer=sgd_opt, loss=cce_loss, metrics=['acc'])
    weights = get_weights()
    model.load_weights(weights)

    return model

def get_weights():
    filename = 'deepface.zip'
    downloaded_file_path = keras.utils.get_file(filename, DOWNLOAD_PATH,
        md5_hash=MD5_HASH, extract=True)
    downloaded_h5_file = path.join(path.dirname(downloaded_file_path),
        path.basename(DOWNLOAD_PATH).rstrip('.zip'))
    return downloaded_h5_file


def create_deepface():
    model = create_classifying_deepface()
    weights = get_weights()
    model.load_weights(weights)
    
    model2 = tf.keras.Sequential()

    # remove last dropout layer and dense layer
    for layer in model.layers[:-2]:
        model2.add(layer)

    sgd_opt = tf.keras.optimizers.SGD(learning_rate=LEARN_RATE, momentum=MOMENTUM)
    cce_loss = tf.keras.losses.CategoricalCrossentropy()

    model2.compile(optimizer=sgd_opt, loss=cce_loss, metrics=['acc'])
    model2.build((None, *IMAGE_SIZE, CHANNELS))
    
    return model2


def get_deepface_rep(filename):

    model = create_deepface()
    im = Image.open(filename)
    im = im.resize(IMAGE_SIZE)
    im = np.array(im, dtype=np.float32)
    output = model.predict(np.expand_dims(im, axis=0))

    return output





