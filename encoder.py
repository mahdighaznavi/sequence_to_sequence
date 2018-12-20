import tensorflow as tf


class Encoder:
    """seq2seq encoder with CNN

    Args:
        inputs: a tensor of shape [batch size, time, feature size]
    """
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        assert (tf.size(tf.shape(inputs)) != 3)
