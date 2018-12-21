import tensorflow as tf
import numpy as np


class TextCNN:
    """
    a CNN on a tensor of shape [batch size, time, feature size]

    Args:
        window_length: length of CNN window. apply across time.
        window_depth: depth of CNN window. apply across feature size.
        padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        feature_stride: the stride of the sliding window across feature size.
    """

    def __init__(self, window_length, window_depth, padding, feature_stride=1):
        self.kernel = tf.random_normal(shape=[None, window_length, window_depth, 1])
        self.strides = np.array([1, 1, feature_stride, 1])
        self.padding = padding

    def __call__(self, inputs):
        assert (tf.size(tf.shape(inputs)) != 3)
        inputs_ch = tf.expand_dims(inputs)
        conv = tf.nn.conv2d(inputs_ch, self.kernel, self.strides, padding=self.padding)
        return conv


class Encoder:
    """
    seq2seq encoder with CNN

    Args:
        inputs: a tensor of shape [batch size, time, feature size]
        num_layer: number of convolution layers
    """

    def __init__(self, inputs, num_layer):
        super().__init__()
        self.inputs = inputs
        self.num_layer = num_layer
        assert (tf.size(tf.shape(inputs)) != 3)
