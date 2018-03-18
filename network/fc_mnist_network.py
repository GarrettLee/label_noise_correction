import network.dnn_network as dnn_network
import network.network_base as network_base
import tensorflow as tf

__author__ = 'garrett_local'


class FcMnistNetwork(dnn_network.DNNNetwork):
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
                 transition_mat=None):
        """
        Initialize all the placeholders.
        :param loss_type: basestring, indicating which type of loss is used,
                    including cross_entropy, backward, backward_t, forward,
                    forward_t.
        :param transition_mat: Numpy mat for transition matrix which describes
                    label noise.
        :param trainable: if the network is trainable.
        """
        if loss_type != 'cross_entropy' and transition_mat is None:
            raise ValueError('transition_mat must be set when using loss '
                             'correction.')
        dnn_network.DNNNetwork.__init__(self, trainable)

        self.loss_type = loss_type
        self.do_summarizing = do_summarizing
        self.x = None
        self.y = None
        self.keep_prob = None
        self.transition_mat = transition_mat


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
        with tf.variable_scope('input'):
            self.x = tf.placeholder('float', shape=[None, 784], name='x')
            self.y = tf.placeholder('float', shape=[None, 10], name='y')
            self.keep_prob = tf.placeholder('float', shape=[], name='keep_prob')
            if ((self.loss_type == 'backward') or
                (self.loss_type == 'backward_t') or
                (self.loss_type == 'forward') or
                (self.loss_type == 'forward_t')):
                self.transition_tensor = tf.Variable(
                    self.transition_mat,
                    dtype=tf.float32,
                    trainable=False
                )
                self.layers['t'] = self.transition_tensor
        self.layers['keep_prob'] = self.keep_prob
        self.layers['x'] = self.x
        self.layers['y'] = self.y


        (self.feed(('x')).
         fc(128, name='fc1', do_summarizing=self.do_summarizing).
         dropout().
         fc(128, name='fc2', do_summarizing=self.do_summarizing).
         dropout().
         fc(10, name='fc3', activation='softmax',
            weights_initializer=tf.random_uniform_initializer(-0.05, 0.05),
            do_summarizing=self.do_summarizing)
         )
        with tf.name_scope('loss'):
            if self.loss_type == 'cross_entropy':
                loss = -tf.reduce_mean(tf.reduce_sum(self.get_output('y') *
                                                     tf.log(self.get_output('fc3')
                                                            + 10e-12),
                                                     reduction_indices=[1]))
                self.layers['loss'] = loss
            elif self.loss_type == 'backward':
                y_trans = tf.transpose(self.get_output('y'), perm=[1,0])
                t_inv = tf.matrix_inverse(self.get_output('t'))
                t_inv_trans = tf.transpose(t_inv, perm=[1,0])
                l_orig = -tf.log(self.get_output('fc3') + 10e-12)
                l_backward_full = tf.matmul(l_orig, t_inv_trans)
                loss = tf.reduce_mean(
                    tf.reduce_sum(tf.matrix_band_part(
                        tf.matmul(l_backward_full, y_trans),
                        0,
                        0), reduction_indices=[1]))
                self.layers['loss'] = loss
            elif self.loss_type == 'forward':
                corrected_pred = tf.matmul(self.get_output('fc3'),
                                           self.get_output('t'))
                loss = -tf.reduce_mean(tf.reduce_sum(self.get_output('y') *
                                                     tf.log(corrected_pred
                                                            + 10e-12),
                                                     reduction_indices=[1]))
                self.layers['corrected_pred'] = corrected_pred
                self.layers['loss'] = loss
            else:
                raise RuntimeError('Incorrect loss function.')
        self.add_summary(loss, name='loss')

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
