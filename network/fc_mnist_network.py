import tensorflow as tf

import network.dnn_network as dnn_network
import network.network_base as network_base
import network.loss_correction_network as loss_correction_network

__author__ = 'garrett_local'


class FcMnistNetwork(dnn_network.DNNNetwork,
                     loss_correction_network.LossCorrectionNetwork):
    """
    Network for Fully connected network on MNIST experiment in paper:
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction
    Approach by Giorgio Patrini etc.

    To use the network:
        feed_dict: if loss_type == 'cross_entropy', placeholder 'x', 'y',
            'keep_prob' are needed to be fed. Otherwise, placeholder 'x', 'y',
            'keep_prob', 't' are needed to be fed. Among them, 'x' is the input
            feature of MNIST datebase, 'y' is the one-hot label for MNIST
            database, 'keep_prob' is the dropout probability, 't' is the label
            transition matrix.
        fetch_list: 'fc3' is the prediction of the network, 'loss' is the loss
            of the network.
    Attributes:
        x: placeholder for input feature of the network. MNIST features. It
            should be of shape [None, 784].
        y: placeholder for input label of the network. MNIST one-hot label.
            It should be of shape [None, 10].
        keep_prob: placeholder for dropout probability. It should be a scalar.
        loss_type: basestring, indicating which type of loss is used, including
            cross_entropy, backward, backward_t, forward, forward_t
        transition_mat: placeholder for transition matrix T.
        others see its supper class.
    """
    def __init__(self, loss_type='cross_entropy',
                 trainable=True, do_summarizing=False,
                 transition_mat=None, keep_prob=0.5):
        loss_correction_network.LossCorrectionNetwork.__init__(self,
                                                               loss_type,
                                                               trainable,
                                                               do_summarizing,
                                                               transition_mat)
        self.keep_prob_to_feed = keep_prob

    @network_base.layer
    def dropout(self, inputs, name):
        """
        A dropout layer using a designed keep probability.
        :param inputs: a tuple with a tensor as its only one element.
        :param name: naem of the layer.
        :return: dropout output, usually followed by an fc layer.
        """
        return dnn_network.DNNNetwork.dropout.\
            _original(self, inputs,
                      self.get_output('keep_prob'),
                      name)

    def setup(self):
        """
        Implementation of the network architecture.
        """
        self.build_input_placeholder()
        with tf.compat.v1.variable_scope('input'):
            self.keep_prob = tf.compat.v1.placeholder('float', shape=[], name='keep_prob')
        self.layers['keep_prob'] = self.keep_prob

        (self.feed(('x')).
         fc(128, name='fc1', do_summarizing=self.do_summarizing).
         dropout().
         fc(128, name='fc2', do_summarizing=self.do_summarizing).
         dropout().
         fc(10, name='fc3', activation='softmax',
            weights_initializer=tf.random_uniform_initializer(-0.05, 0.05),
            do_summarizing=self.do_summarizing)
         )
        self.build_loss()

    def build_input_placeholder(self):
        with tf.compat.v1.variable_scope('input'):
            self.x = tf.compat.v1.placeholder('float', shape=[None, 784], name='x')
            self.y = tf.compat.v1.placeholder('float', shape=[None, 10], name='y')
        self.layers['x'] = self.x
        self.layers['y'] = self.y

    def generate_feed_dict_for_training(self, fed_data):
        x = fed_data[0]
        y = fed_data[1]
        return {self.get_placeholder_x(): x, self.get_placeholder_y(): y,
                self.get_placeholder_keep_prob(): self.keep_prob_to_feed}

    def generate_feed_dict_for_testing(self, fed_data):
        x = fed_data[0]
        return {self.get_placeholder_x(): x,
                self.get_placeholder_keep_prob(): 1.0}

    def get_placeholder_x(self):
        """
        This should return a placeholder for feature feeding.
        :return: a placeholder for feature feeding.
        """
        return self.get_output('x')

    def get_placeholder_y(self):
        """
        This should return a placeholder for label feeding.
        :return: a placeholder for label feeding.
        """
        return self.get_output('y')

    def get_placeholder_keep_prob(self):
        """
        This should return a
        placeholder for dropout probability.
        :return: a placeholder for dropout probability.
        """
        return self.get_output('keep_prob')

    def get_tensor_prediction(self):
        """
        This should return a tensor for prediction.
        :return: a placeholder for prediction.
        """
        return self.get_output('fc3')

    def get_tensor_loss(self):
        """
        This should return a tensor for loss.
        :return: a placeholder for loss.
        """
        return self.get_output('loss')
