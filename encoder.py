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

    def __init__(self, window_length, window_depth, output_channel, padding="SAME", feature_stride=1):
        self.kernel = tf.random_normal(shape=[window_length, window_depth, 1, output_channel])
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
        num_layer: number of convolution layers
        window_length: length of CNN window. apply across time.
        window_depth: depth of CNN window. apply across feature size.
        output_channel = size of output channel
        feature_stride: the stride of the sliding window across feature size.
        padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
    """

    def __init__(self, num_layer, window_length, window_depth, feature_stride, output_channel, padding="SAME"):
        self.output_channel = output_channel
        self.num_layer = num_layer
        self.window_length = window_length
        self.window_depth = window_depth
        self.padding = padding
        self.feature_stride = feature_stride

    def __call__(self, inputs):
        """

        :param inputs: a batch of sequence of tokens with shape [batch size, time, feature size]
        :return: result of convolution over input
        """

        assert (tf.size(tf.shape(inputs)) != 3)
        next_input = inputs
        cnn = TextCNN(self.window_length, self.window_depth, self.output_channel, self.padding, self.feature_stride)
        for i in range(0, self.num_layer):
            next_input = cnn(next_input)

        return next_input

