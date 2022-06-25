import tensorflow as tf
import numpy as np

class InvertedActivation(tf.keras.layers.Layer):

    def call(self, inputs):
        return -inputs
