import tensorflow as tf
import network.network_base as network_base

__author__ = 'garrett_local'


class DNNNetwork(network_base.NetworkBase):
    """Implementing common network layers."""

    def __init__(self, inputs, trainable=True):
        network_base.NetworkBase.__init__(self, trainable)

    @network_base.layer
    def fc(self, inputs, channel_out, name='fc',
           activation='relu', weights_initializer=None, trainable=True):
        """
        FC layer.
        :param inputs: a tuple with only one element. The element must be a
                    tensor with shape [BATCH_SIZE, CHANNEL_IN]
        :param channel_out: output channel.
        :param name: name of the layer.
        :param activation: a basestring. Name of a implemented activation.
        :param weights_initializer: a TensorFlow initializer for weights
                    initialization.
        :param trainable: bool.
        :return: a tensor of fc output.
        """
        assert len(inputs) == 1, \
            'fc layer accept only one tensor for input.'
        assert inputs[0].shape.ndims == 2, \
            'Shape of fc input must be [BATCH_SIZE, CHANNEL_IN].'
        if weights_initializer is None:
            weights_initializer = \
                tf.contrib.layers.variance_scaling_initializer(
                    factor=2.0,
                    mode='FAN_IN',
                    uniform=False
                ) # Delving deep into rectifier by He.
        biases_initializer = tf.constant_initializer(0.0)
        channel_in = inputs[0].shape[1]
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('weights') as scope:
                weghts = tf.get_variable('W', [channel_in, channel_out],
                                         initializer=weights_initializer,
                                         trainable=trainable)
            with tf.variable_scope('biases') as scope:
                biases = tf.get_variable('b', [1, channel_out],
                                         initializer=biases_initializer,
                                         trainable=trainable)
            if activation is not None:
                activation_op = self._parse_activation(activation)
                fc = activation_op(tf.matmul(inputs[0], weghts) + biases)
            else:
                fc = tf.matmul(inputs[0], weghts) + biases
        return fc

    @network_base.layer
    def dropout(self, inputs, keep_prob, name):
        """
        Dropout layer. Should be used with input of an fc layer.
        :param inputs: a tuple with a tensor as its only one element.
        :param keep_prob: a scalar tensor.
        :param name: naem of the layer.
        :return: dropout output, usually followed by an fc layer.
        """
        assert len(inputs) == 1, \
            'Dropout layer accept only one tensor for input.'
        return tf.nn.dropout(inputs[0], keep_prob, name=name)

    @staticmethod
    def _parse_activation(activation):
        """
        Return a TensorFlow activation op according to its name.
        :param activation: basestring. Name of a implemented activation.
        :return: TensorFlow op.
        """
        if activation == 'relu':
            return tf.nn.relu
        if activation == 'tanh':
            return tf.nn.tanh
        if activation == 'sigmoid':
            return tf.nn.sigmoid
        if activation == 'softmax':
            return tf.nn.softmax
        raise Exception, 'Unknown activation type.'