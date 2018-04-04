import os
import re
import tensorflow as tf
import helper.cfg_tools as cfg_tools
import helper.file_manager as file_manager
import network.network_base as network_base

__author__ = 'garrett_local'


class TrainerBase(object):
    """
    A trainer class is a class interacting with subclass of 
    network.network_base.NetworkBase.
    """
    def __init__(self, cfg_path=None, do_summarizing=False, summary_path=None):
        if cfg_path is None:
            cfg = {}
            cfg['optimization'] = {}
            cfg['optimization']['algorithm'] = 'AdaGrad'
            cfg['optimization']['learning_rate'] = '0.01'
            cfg['optimization']['delta'] = '10e-6'
            cfg['optimization']['epoch'] = '40'

            cfg['batch_size'] = {}
            cfg['batch_size']['batch_size'] = '128'

            self._cfg = cfg
            self._save_cfg('./cfg/experiments/imdb/example')
        else:
            cfg = cfg_tools.read_cfg_file(cfg_path)
            self._cfg = cfg

        self._session = None
        self._network = None
        self._do_summarizing = do_summarizing
        self._summary_writer = None
        self._summaries = None
        self._iter = 0
        self._train_op = None
        self._summary_path = summary_path
        if self._do_summarizing:
            if self._summary_path is None:
                raise ValueError(
                    'summary_path must be set if do_summarizing is ture.'
                )

    def __del__(self):
        if self._session is not None:
            self._session.close()

    @property
    def iter(self):
        """
        Training step.
        :return: an int.
        """
        return self._iter

    def get_batch_size(self):
        return int(self._cfg['batch_size']['batch_size'])

    def get_train_op(self):
        assert self._network is not None, 'Network not defined.'
        if self._cfg['optimization']['algorithm'] == 'AdaGrad':
            optimizer = tf.train.AdagradOptimizer(
                float(self._cfg['optimization']['learning_rate']),
                initial_accumulator_value=float(self._cfg['optimization']['delta']))
        return optimizer.apply_gradients(
            self._network.gradient(self._network.get_tensor_loss(),
                                   do_summarizing=self._do_summarizing)
        )

    def setup_network(self, network):
        assert issubclass(network.__class__, network_base.NetworkBase), \
            'network must be a subclass of NetworkBase class.'
        self._network = network
        graph = tf.Graph()
        with graph.as_default():
            network.setup()
            self._train_op = self.get_train_op()
            self._summaries = network.get_summaries()
            init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=graph, config=config)
        if self._do_summarizing:
            self._summary_writer = tf.summary.FileWriter(
                self._summary_path,
                self._session.graph
            )
        self._session.run(init)

    def train(self, fed_data):
        """
        Train a step. Notice that fed_data must have the same format as the
        argument of function generate_feed_dict_for_training implemented in
        the class of the network object.
        :param fed_data: should has the same format and meaning with the
                    argument implemented in function
                    generate_feed_dict_for_training implemented in the class of
                    network object.
        :return: a float. The loss.
        """
        if self._do_summarizing:
            _, summary_result, loss = self._session.run(
                [self._train_op, self._summaries, self._network.get_tensor_loss()],
                feed_dict=self._network.generate_feed_dict_for_training(fed_data)
            )
            self._summary_writer.add_summary(summary_result, self._iter)
            self._summary_writer.flush()
        else:
            _, loss = self._session.run(
                [self._train_op, self._network.get_tensor_loss()],
                feed_dict=self._network.generate_feed_dict_for_training(fed_data)
            )
        self._iter += 1
        return loss

    def test(self, fed_data):
        """
        test a step. Notice that fed_data must have the same format as the
        argument of function generate_feed_dict_for_testing implemented in
        the class of the network object.
        :param fed_data: should has the same format and meaning with the
                    argument implemented in function
                    generate_feed_dict_for_testing implemented in the class of
                    network object.
        :return: the prediction.
        """
        return self._session.run(
            self._network.get_tensor_prediction(),
            feed_dict=self._network.generate_feed_dict_for_testing(fed_data)
        )

    def validate(self, fed_data):
        """
        validate a step. Notice that fed_data must have the same format as the
        argument of function generate_feed_dict_for_training implemented in
        the class of the network object. What is different from function train
        is that there is no summarizing and no variable being updated.

        :param fed_data: should has the same format and meaning with the
                    argument implemented in function
                    generate_feed_dict_for_training implemented in the class of
                    network object.
        :return: a float. The loss.
        """
        loss = self._session.run(
            self._network.get_tensor_loss(),
            feed_dict=self._network.generate_feed_dict_for_training(fed_data)
        )
        return loss


    def get_max_epoch(self):
        """
        Get the max epoch from self._cfg.
        :return: an int. Indicating how many epoch should be performed.
        """
        return int(self._cfg['optimization']['epoch'])

    def save_model(self, path, save_type='tensorflow_save'):
        """
        Save the model. Model can be saved in two modes: tensorflow_save,
        npy_save. The difference is that under npy_save, saved data can easily
        be shared with another network. Model will be saved with its name
        ending with current training step (self.iter).

        :param path: a basestring. Indicating where to save.
        :param save_type: could be 'tensorflow_save' or 'npy_save'. Indicating
                    how to save the model.
        :return: None.
        """
        assert self._session is not None, ('Execute function self.setup_network '
                                         'before saving.')
        file_manager.create_if_not_exist(path)
        if save_type == 'tensorflow_save':
            with self._session.graph.as_default():
                saver = tf.train.Saver()
                saver.save(
                    self._session,
                    os.path.join(path, 'model.ckpt'),
                    self._iter
                )
        if save_type == 'npy_save':
            self._network.save_network_to_npy(
                os.path.join(path,
                             'model_{0}.npy'.format(self._iter)),
                self._session
            )

    def load_model(self, path, iter=None, load_type='tensorflow_save'):
        """
        Load model. Model can be loaded in two modes: tensorflow_save,
        npy_save. In mode npy_save, Model will be saved with its name
        ending with current training step (self.iter).

        :param path: a basestring. Indicating where to save.
        :param save_type: could be 'tensorflow_save' or 'npy_save'. Indicating
                    how to save the model.
        :return: None.
        """
        if iter is None:
            if load_type == 'tensorflow_save':
                last_modified_model_name = file_manager.last_modified(
                    os.path.join(path,
                                 'model.ckpt-*.data*')
                )
                if last_modified_model_name is None:  # path doesn't exist.
                    return None
                iter = re.findall(
                    '\d+',
                    os.path.basename(last_modified_model_name)
                )[0]
                with self._session.graph.as_default():
                    saver = tf.train.Saver()
                    try:
                        saver.restore(
                            self._session,
                            os.path.join(path,
                                         'model.ckpt-{0}'.format(iter))
                        )
                    except (tf.errors.NotFoundError,
                            tf.errors.InvalidArgumentError):  # File doesn't exist.
                        return None

            elif load_type == 'npy_save':
                last_modified_model_name = file_manager.last_modified(
                    os.path.join(path,
                                 'model_*.npy')
                )
                if last_modified_model_name is None:  # path doesn't exist.
                    return None
                self._network.restore_network_from_npy(
                    last_modified_model_name,
                    self._session
                )
                iter = re.findall(
                    '\d+',
                    os.path.basename(last_modified_model_name)
                )[0]
            else:
                raise ValueError('load_type is illegal.')
        else:
            if load_type == 'tensorflow_save':
                with self._session.graph.as_default():
                    saver = tf.train.Saver()
                    try:
                        saver.restore(
                            self._session,
                            os.path.join(
                                path,
                                'model.ckpt-{0}'.format(int(iter))
                            )
                        )
                    except (tf.errors.NotFoundError,
                            tf.errors.InvalidArgumentError):  # File doesn't exist.
                        return None
            elif load_type == 'npy_save':
                try:
                    self._network.restore_network_from_npy(
                        os.path.join(
                            path,
                            'model_{0}.npy'.format(int(iter))
                        ),
                        self._session
                    )
                except IOError:  # File doesn't exist.
                    return None
            else:
                raise ValueError('load_type is illegal.')
        if iter is not None:
            self._iter = iter
            return iter
        else:
            return None

    def _save_cfg(self, path):
        cfg_tools.write_cfg_file(path, self._cfg)
