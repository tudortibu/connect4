import utility as util
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import warnings


def main():


    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(( 6, 7, 1), input_shape=(42,1)),
        tf.keras.layers.Conv2D(256,5,
                               input_shape=(1, 6, 7, 1)),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(450, activation=tf.nn.relu),
        tf.keras.layers.Dense(210, activation=tf.nn.relu),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    model.summary()
main()