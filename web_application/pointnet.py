from os import listdir
from os.path import join, isfile
import numpy as np #1.16.4 otherwise futurewarning tensorflow
import tensorflow as tf
import csv
from sklearn.model_selection import LeaveOneOut

NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_POINTS = 510


def conv_bn(x, filters):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.keras.layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = tf.keras.layers.Dense(filters)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.keras.layers.Activation("relu")(x)


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
        
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2,2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


def tnet(inputs, num_features):
    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
    
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = tf.keras.layers.Dense(
        num_features * num_features, 
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg)(x)
    
    feat_T = tf.keras.layers.Reshape((num_features, num_features))(x)
    
    return tf.keras.layers.Dot(axes=(2,1))([inputs, feat_T])


def generate_model():

    inputs = tf.keras.Input(shape=(NUM_POINTS, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    x = dense_bn(x, 128)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=["sparse_categorical_accuracy"])
    
    return model


