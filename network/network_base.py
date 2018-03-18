import abc
import tensorflow as tf
import numpy as np

__author__ = 'garrett_local'


def include_original(dec):
    """
    A decorator for decorator. It saves original function as a attribute of
    the returned function.
    :param dec: decorated decorator.
    :return: decorated function.
    """

    def meta_decorator(f):
        decorated = dec(f)
        decorated ._original = f
        return decorated

    return meta_decorator


@include_original
def layer(op):
    """
    A decorator for network op.

    Via this decorator:
    1. Output of op will be stored automatically.
    2. TensorBoard summaries will be perform automatically.
    3. Enable users to build a network in a chain.
    4. Can save network to npy file and restore network from npy file.
    :param op: network op.
    :return: decorated op.
    """

    def layer_decorated(self, *args, **kwargs):

        name = kwargs.setdefault('name',
                                 self._get_unique_name(op.__name__.strip('_')))
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer {0}'.
                               format(name))
        else:
            layer_input = list(self.inputs)

        if kwargs.has_key('do_summarizing'):
            do_summarizing = kwargs.pop('do_summarizing')
        else:
            do_summarizing = False

        layer_output = op(self, layer_input, *args, **kwargs)
        self.layers[name] = layer_output
        self.feed((name, ))


        if do_summarizing:
            variables =  self.trainable_variables(name)
            if len(variables) != 0:
                summaries = []
                for variable in variables:
                    summaries.append(tf.summary.histogram("{0}/hist".
                                                          format(variable.name),
                                                          variable))
                summaries = tf.summary.merge(summaries)
                self.variable_summaries[name] = summaries
        return self # return self, so network can be build in form of chains.
    return layer_decorated


class NetworkBase(object):
    """ Implementation of higher class op with TensorFlow.

    With this class, summaries can be processed automatically, and network can
    be build in a clearer way(in a chain).

    To use the class, one must implement ops in a specific way:
        def op_example(inputs, ...)
            ...
            return outpus
    Some condition must be satisfied:
    1. Inputs and outputs of an op must be tuples.
    2. Trainable variables created inside the op must be under a variable scope
        named by the given name.
    3. If there is any trainable variables which is weight in an op, create them
        under the variable scope given_name/weights. Then we can get its weight
        with function get_weight_of_layer later.

    Build a network in a chain:
        (network_base.feed(inputs1).
         op1(inputs2, ...).
         op2(inputs3, ...).
         ...
        )

    Attributes:
        inputs: a tuple storing inputs for next op.
        layers: a dict storing outputs for each layer.
        variable_summaries: A tuple of dict storing summaries of trainable
            variables for each layer.
        gradient_summaries: A tuple of dict storing summaries of gradient of
            trainable variables for each layer.
        trainable: bool variable indicating if the network is trainable.
    """

    def __init__(self, trainable=True):
        """
        Init.
        :param inputs: Input of a network. Should be a tuple.
        :param trainable: if trainable.
        """
        self.layers = {}
        self.variable_summaries = {}
        self.gradient_summaries = {}
        self.trainable = trainable

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
        placeholder for label feeding.
        :return: a placeholder for label feeding.
        """
        pass

    @abc.abstractmethod
    def get_placeholder_keep_prob(self):
        """
        An abstract function must be implemented. This should return a
        placeholder for dropout probability.
        :return: a placeholder for dropout probability.
        """
        pass

    @abc.abstractmethod
    def get_tensor_prediction(self):
        """
        An abstract function must be implemented. This should return a
        tensor for prediction.
        :return: a placeholder for prediction.
        """
        pass

    @abc.abstractmethod
    def get_tensor_loss(self):
        """
        An abstract function must be implemented. This should return a
        tensor for loss.
        :return: a placeholder for loss.
        """
        pass

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def feed(self, to_feed):
        """
        Reset self.inputs for the next op.
        :param to_feed: a tuple of basestring which must be key of dict self.layers.
        :return: The object itself (return self).
        """
        assert len(to_feed) != 0, 'Input must not be empty.'
        self.inputs = []
        return self.append_inputs(to_feed)

    def add_summary(self, tensor, name):
        """
        Add a Tensor into summary.
        :param tensor: TensorFlow Tensor.
        :param name: name.
        """
        ndims = tensor.shape.ndims
        if ndims == 0:
            self.variable_summaries[name] = tf.summary.scalar(name, tensor)
        else:
            self.variable_summaries[name] = tf.summary.histogram('{0}/hist'.
                                                                 format(name),
                                                                 tensor)

    def append_inputs(self, to_append):
        """
        Append inputs in self.inputs.
        :param to_append: a tuple of basestring which must be key of dict self.layers.
        :return: The object itself (return self).
        """
        assert len(to_append) != 0, 'Input must not be empty.'
        for layer in to_append:
            try:
                self.inputs.append(self.layers[layer])
                print layer
            except KeyError:
                print self.layers.keys()
                raise KeyError('Unknown layer name fed: {0}'.format(layer))
        return self

    def get_output(self, layer):
        """
        Get outputs of a specified layer.
        :param layer: a basestring which is one of the keys of self.layers.
        :return: a TensorFlow tensor.
        """
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name to be fetched: {0}'.format(layer))
        return layer

    def get_weight_of_layer(self, layer_name):
        """
        Get weights created in a layer.
        :param layer_name: a basestring which is one of the keys of self.layers.
        :return: a dict of TensorFlow Tensor.
        """
        return self.trainable_variables('{0}/{1}'.format(layer_name, 'weights'))

    def _get_unique_name(self, base):
        """
        Add a id number as postfix to a string, make the string unique.
        :param base: base of the string.
        :return: a unique new string.
        """
        id = sum(t.startswith(base + '_') for t, _ in self.layers.items()) + 1
        return '{0}_{1}'.format(base, id)

    def gradient(self, y, tvars=None, do_summarizing=False):
        """
        Get the gradients of y above multiple variables.
        :param y: a TensorFlow Tensor. Dependent variable.
        :param tvars: a list of TensorFlow Tensor. All must be arguments of y.
        :param do_summarizing: If ture, summaries will be appended into
                    self.gradient_summaries.
        :return a list of Tensors. Each element of the list is a tuple of two
                    Tensor. The first Tensor in the list is gradient and the
                    second Tensor is its corresponding trainable variable.
        """
        if tvars is None:
            tvars = tf.trainable_variables()
        grads = tf.gradients(y, tvars)
        if do_summarizing:
            for idx in range(len(tvars)):
                name = '{0}_{1}'.format(y.name, tvars[idx].name)
                # Prevent from duplicating.
                if not self.gradient_summaries.has_key(name):
                    self.gradient_summaries[name] = \
                        tf.summary.histogram('grad/{0}'.format(name),
                                             grads[idx])
        return zip(grads, tvars)

    def get_summaries(self):
        """
        Merge all summaries and return.
        :return:
        """
        return tf.summary.merge_all()

    def save_network_to_npy(self, data_path, session):
        """
        Save network trainable variable to a numpy npy file.

        Created npy file should be restore with function
        self.restore_network_from_npy.
        :param data_path: basestring. Path to save the npy file.
        :param session: TensorFlow Session.
        """
        tvars = self.trainable_variables()
        values = session.run(tvars)
        values_dicts = []
        for idx, value in enumerate(values):
            values_dict = {}
            name = tvars[idx].name.split('/')[-1]
            name_len = len(name)
            scope = tvars[idx].name[:-1 * name_len - 1]
            values_dict['scope'] = scope
            values_dict['name'] = name.split(':')[0]
            values_dict['value'] = value
            values_dicts.append(values_dict)
        np.save(data_path, values_dicts)

    def restore_network_from_npy(self, data_path, session,
                                 ignore_missing=False):
        """
        Restore network trainable variable from a numpy npy file.

        This npy file must has been created with self.save_network_to_file
        function.
        :param data_path: basestring. Path to npy file.
        :param session: Tensorflow Session.
        :param ignore_missing: if it is set true, variable missing will be
                    ignore.
        """
        data_dicts = np.load(data_path)
        for data_dict in data_dicts:
            with tf.variable_scope(data_dict['scope'], reuse=True):
                try:
                    var = tf.get_variable(data_dict['name'])
                    session.run(var.assign(data_dict['value']))
                    print 'assign pretrain model to {0}/{1}'.\
                        format(data_dict['scope'], data_dict['name'])
                except ValueError:
                    print 'ignore {0}/{1}'.format(data_dict['scope'],
                                                  data_dict['name'])
                    if not ignore_missing:
                        raise

    def trainable_variables(self, scope=None):
        """
        Get trainable variables in a specified variable scope.
        :param scope: name of a variable scope, can also be None, in which case
                    all trainable variables in the whole graph will be return.
        :return: trainable variables in a specified variable scope. It is a list
                    of dict. each dict corresponds to a trainable variable.
        """
        if scope is None:
            return tf.trainable_variables() # return all variables.
        else:
            with tf.variable_scope(scope) as scope:
                return scope.trainable_variables()