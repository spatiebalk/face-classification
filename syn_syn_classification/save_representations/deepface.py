#!/usr/bin/env python
# coding: utf-8

# # Implementation of DeepFace using Keras
# In this notebook a [Keras implementation](https://github.com/swghosh/DeepFace) from Github of Facebook's [DeepFace](https://research.fb.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/) is loaded and used. The weights are trained on the publicly available VGG dataset
import keras
import numpy as np
from os import path
from os import listdir
from os.path import isfile, join
from PIL import Image 
from keras.models import Model
import tensorflow
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

def create_classifying_deepface(image_size=IMAGE_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES, learn_rate=LEARN_RATE, momentum=MOMENTUM):
    """
    Deep CNN architecture primarily for Face Recognition,
    Face Verification and Face Representation (feature extraction) purposes
    "DeepFace: Closing the Gap to Human-Level Performance in Face Verification"
    CNN architecture proposed by Taigman et al. (CVPR 2014)
    """

    wt_init = keras.initializers.RandomNormal(mean=0, stddev=0.01)
    bias_init = keras.initializers.Constant(value=0.5)

    """
    Construct certain functions 
    for using some common parameters
    with network layers
    """
    def conv2d_layer(**args):
        return keras.layers.Conv2D(**args, 
            kernel_initializer=wt_init, 
            bias_initializer=bias_init,
            activation=keras.activations.relu)
    def lc2d_layer(**args):
        return keras.layers.LocallyConnected2D(**args, 
            kernel_initializer=wt_init, 
            bias_initializer=bias_init,
            activation=keras.activations.relu)
    def dense_layer(**args):
        return keras.layers.Dense(**args, 
            kernel_initializer=wt_init, 
            bias_initializer=bias_init)

    """
    Create the network using
    tf.keras.layers.Layer(s)
    """
    deepface = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(*image_size, channels), name='I0'),
        conv2d_layer(filters=32, kernel_size=11, name='C1'),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same',  name='M2'),
        conv2d_layer(filters=16, kernel_size=9, name='C3'),
        lc2d_layer(filters=16, kernel_size=9, name='L4'),
        lc2d_layer(filters=16, kernel_size=7, strides=2, name='L5'),
        lc2d_layer(filters=16, kernel_size=5, name='L6'),
        keras.layers.Flatten(name='F0'),
        dense_layer(units=4096, activation=keras.activations.relu, name='F7'),
        keras.layers.Dropout(rate=0.5, name='D0'),
        dense_layer(units=num_classes, activation=keras.activations.softmax, name='F8')
    ], name='DeepFace')
    # deepface.summary()

    """
    A tf.keras.optimizers.SGD will
    be used for training,
    and compile the model
    """
    sgd_opt = keras.optimizers.SGD(lr=learn_rate, momentum=momentum)
    cce_loss = keras.losses.categorical_crossentropy

    deepface.compile(optimizer=sgd_opt, loss=cce_loss, metrics=['accuracy'])
    
    return deepface


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
    model2 = Model(model.input, model.layers[-2].output)
    model2.summary()
    return model2


def deepface_reps(GENERAL_DIR, syn_name):

    model = create_deepface()
    syn_rep, ID_rep = [], []

    syn_dir = GENERAL_DIR + "\\{}\{}-patients".format(syn_name, syn_name)
    ID_dir = GENERAL_DIR + "\\{}\{}-selected-ID-controls".format(syn_name, syn_name)

    # get list of filenames
    files_syn = [f for f in listdir(syn_dir) if (isfile(join(syn_dir, f)))and syn_name in f] #"kdv" for KDVS
    files_ID = [f for f in listdir(ID_dir) if (isfile(join(ID_dir, f))) and ".JPG" in f or ".jpg" in f]

    # for each kdv image save deepface rep as list:
    for filename in tqdm(files_syn):
        im = Image.open(join(syn_dir, filename))
        im = im.resize(IMAGE_SIZE)
        output = model.predict(np.expand_dims(im, axis=0))
        syn_rep.append([filename] + output[0].tolist())  


    # for each ID image save deepface rep as list:
    for filename in tqdm(files_ID):
        im = Image.open(join(ID_dir, filename))
        im = im.resize(IMAGE_SIZE)
        output = model.predict(np.expand_dims(im, axis=0))
        ID_rep.append([filename] + output[0].tolist())

    print("Syn_reps: {}, ID_reps: {}".format(len(syn_rep), len(ID_rep)))

    # location to save representation
    csv_file_syn = GENERAL_DIR + "\\{}\\representations\\{}-patients-deepface.csv".format(syn_name, syn_name)
    csv_file_ID = GENERAL_DIR + "\\{}\\representations\\ID-controls-deepface.csv".format(syn_name)
    
    # save representation of kdv patients
    with open(csv_file_syn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(syn_rep)

    # save representation of ID controls
    with open(csv_file_ID, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(ID_rep)

    print("Done with saving all deepface representations for {}.".format(syn_name))




