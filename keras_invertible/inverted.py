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

""" Invert bias from master layer """
class InvertedBias(tf.keras.layers.Layer):
    def __init__(self, master_layer):
        super(InvertedBias, self).__init__()
        self.master_layer = master_layer

    def call(self, inputs):  # Defines the computation from inputs to outputs
        b = self.master_layer._trainable_weights[1]
        return inputs - b

""" Force layer to remain the Penrose pseudo-inverse of the master layer """
class InvertedDensePI(tf.keras.layers.Layer):
    def __init__(self, master_layer, **kwargs):
        super().__init__(**kwargs)
        self.master_layer = master_layer

    def build(self, input_shape):
        self.W = tf.linalg.pinv(self.master_layer._trainable_weights[0])
        self.b = self.master_layer._trainable_weights[1]
        # do not train weights or bias from master_layer, they are read-only
        self.params = []
        
    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs - self.master_layer._trainable_weights[1], tf.linalg.pinv(self.master_layer._trainable_weights[0]))

    """ Given a master layer, invert bias then transpose weights """
class InvertedDensePI2(tf.keras.layers.Layer):
    def __init__(self, master_layer, **kwargs):
        super().__init__(**kwargs)
        self.master_layer = master_layer

    def build(self, input_shape):
        # do not train weights or bias from master_layer, they are read-only
        self.params = []
        
    def call(self, inputs):  # Defines the computation from inputs to outputs
        W = self.master_layer._trainable_weights[0]
        b = self.master_layer._trainable_weights[1]
        w = tf.linalg.pinv(W)
        return tf.matmul(inputs - b, w)

class InvertedPReLU(tf.keras.layers.Layer):
    def __init__(self, master_layer):
        super(InvertedPReLU, self).__init__()
        self.master_layer = master_layer

    def build(self, input_shape):
        # do not train weights or bias from master_layer, they are read-only
        self.params = []

    def call(self, inputs):
        alpha = self.master_layer.alpha
        pos = keras.backend.relu(-inputs)
        neg = -alpha * keras.backend.relu(inputs)
        return -(pos + neg)

class InvertedLeakyReLU(tf.keras.layers.Layer):

    def __init__(self, alpha=0.3, **kwargs):
        super().__init__(**kwargs)
        if alpha is None:
            raise ValueError(
                "The alpha value of a Leaky ReLU layer cannot be None, "
                f"Expecting a float. Received: {alpha}"
            )
        self.supports_masking = True
        self.alpha = tf.keras.backend.cast_to_floatx(alpha)

    def call(self, inputs):
        return -tf.keras.backend.relu(-inputs, alpha=self.alpha)

    def get_config(self):
        config = {"alpha": float(self.alpha)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MonotonicPReLU(tf.keras.layers.Layer):
    """Monotonic Parametric Rectified Linear Unit.
    It follows:
    ```
      f(x) = alpha * x for x < 0
      f(x) = x for x >= 0
    ```
    where `alpha` is a learned array with the same shape as x.
    To achieve monotonicity, force learned alpha params above 0 via abs().
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as the input.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.alpha = self.add_weight(
            shape=param_shape,
            name="alpha",
        )
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes={})
        self.built = True

    def call(self, inputs):
        pos = tf.keras.backend.relu(inputs)
        neg = -tf.math.abs(self.alpha) * tf.keras.backend.relu(-inputs)
        return pos + neg

    def compute_output_shape(self, input_shape):
        return input_shape
