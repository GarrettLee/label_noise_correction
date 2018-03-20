import abc
import tensorflow as tf

import network.network_base as network_base

__author__ = 'garrett_local'


class LossCorrectionNetwork(network_base.NetworkBase):
    def __init__(self, loss_type='cross_entropy', trainable=True,
                 do_summarizing=False, transition_mat=None):
        """
        Initialize all the placeholders.
        :param loss_type: basestring, indicating which type of loss is used,
                    including cross_entropy, backward, backward_t, forward,
                    forward_t.
        :param transition_mat: Numpy mat for transition matrix which describes
                    label noise.
        :param trainable: if the network is trainable.
        """
        network_base.NetworkBase.__init__(self, trainable)
        if loss_type != 'cross_entropy' and transition_mat is None:
            raise ValueError('transition_mat must be set when using loss '
                             'correction.')
        self.loss_type = loss_type
        self.do_summarizing = do_summarizing
        self.x = None
        self.y = None
        self.keep_prob = None
        self.transition_mat = transition_mat
        self.transition_tensor = None

    @abc.abstractmethod
    def get_placeholder_x(self):
        """
        An abstract function must be implemented. This should return a
        placeholder for feature feeding.
        :return: a placeholder for feature feeding.
        """
        pass

    @abc.abstractmethod
    def get_placeholder_y(self):
        """
        An abstract function must be implemented. This should return a
        placeholder for label feeding. y should strictly be of shape
        [batch_size, classes_number].
        :return: a placeholder for label feeding.
        """
        pass

    def build_input_placeholder(self):
        with tf.variable_scope('input'):
            self.x = tf.placeholder('float', shape=[None, 784], name='x')
            self.y = tf.placeholder('float', shape=[None, 10], name='y')
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
        self.layers['x'] = self.x
        self.layers['y'] = self.y

    def build_loss(self):
        with tf.name_scope('loss'):
            if self.loss_type == 'cross_entropy':
                loss = -tf.reduce_mean(tf.reduce_sum(
                    self.get_output('y') * tf.log(self.get_tensor_prediction()
                                                  + 10e-12),
                    reduction_indices=[1]
                ))
                self.layers['loss'] = loss
            elif self.loss_type == 'backward':
                y_trans = tf.transpose(self.get_output('y'), perm=[1,0])
                t_inv = tf.matrix_inverse(self.get_output('t'))
                t_inv_trans = tf.transpose(t_inv, perm=[1,0])
                l_orig = -tf.log(self.get_tensor_prediction() + 10e-12)
                l_backward_full = tf.matmul(l_orig, t_inv_trans)
                loss = tf.reduce_mean(
                    tf.reduce_sum(tf.matrix_band_part(
                        tf.matmul(l_backward_full, y_trans),
                        0,
                        0), reduction_indices=[1]))
                self.layers['loss'] = loss
            elif self.loss_type == 'forward':
                corrected_pred = tf.matmul(self.get_tensor_prediction(),
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