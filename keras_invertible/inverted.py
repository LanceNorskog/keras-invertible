import tensorflow as tf
import numpy as np

""" Add after any activation to invert it """
class InvertedActivation(tf.keras.layers.Layer):

    def call(self, inputs):
        return -inputs

""" Given a master layer, invert bias then transpose weights """
class InvertedDense(tf.keras.layers.Layer):
    def __init__(self, master_layer):
        super(InvertedDense, self).__init__()
        self.master_layer = master_layer

    def build(self, input_shape):
        # do not train weights or bias from master_layer, they are read-only
        self.params = []
        
    def call(self, inputs):  # Defines the computation from inputs to outputs
        W = self.master_layer._trainable_weights[0]
        b = self.master_layer._trainable_weights[1]
        w = tf.transpose(W)
        return tf.matmul(inputs - b, w)

    """ Given a master layer, invert bias then transpose weights """
class InvertedDensePI(tf.keras.layers.Layer):
    def __init__(self, master_layer):
        super(InvertedDensePI, self).__init__()
        self.master_layer = master_layer

    def build(self, input_shape):
        # do not train weights or bias from master_layer, they are read-only
        self.params = []
        
    def call(self, inputs):  # Defines the computation from inputs to outputs
        (W, b) = self.master_layer.get_weights()
        print(type(W))
        print('W.shape', W.shape)
        print('b.shape', b.shape)
        PI = np.linalg.pinv(W)
#         W = self.master_layer._trainable_weights[0]
#         b = self.master_layer._trainable_weights[1]
#         w = tf.transpose(W)
        return tf.matmul(inputs - b, w)


class InvertedPReLU(layers.Layer):
    def __init__(self, master_layer):
        super(InvertedPReLU, self).__init__()
        self.master_layer = master_layer

    def build(self, input_shape):
        # do not train weights or bias from master_layer, they are read-only
        self.params = []

    def call(self, inputs):
        alpha = 1/(self.master_layer.alpha + 0.00001)
        pos = keras.backend.relu(inputs)
        neg = -alpha * keras.backend.relu(-inputs)
        return pos + neg
