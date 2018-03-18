import os
import re
import tensorflow as tf
import helper.cfg_tools as cfg_tools
import helper.file_manager as file_manager
import network.network_base as network_base

__author__ = 'garrett_local'


class TrainerBase(object):

    def __init__(self, cfg_path=None, do_summarizing=False, summary_path=None):
        if cfg_path is None:
            cfg = {}
            cfg['optimization'] = {}
            cfg['optimization']['algorithm'] = 'AdaGrad'
            cfg['optimization']['learning_rate'] = '0.01'
            cfg['optimization']['delta'] = '10e-6'
            cfg['optimization']['epoch'] = '40'

            cfg['dropout'] = {}
            cfg['dropout']['probability'] = '0.5'

            cfg['batch_size'] = {}
            cfg['batch_size']['batch_size'] = '128'

            self.cfg = cfg
            self._save_cfg('./cfg/experiments/mnist/example')
        else:
            cfg = cfg_tools.read_cfg_file(cfg_path)
            self.cfg = cfg

        self.session = None
        self.network = None
        self.do_summarizing = do_summarizing
        self.summary_writer = None
        self.summaries = None
        self._iter = 0
        self.train_op = None
        self.summary_path = summary_path
        if self.do_summarizing:
            if self.summary_path is None:
                raise ValueError(
                    'summary_path must be set if do_summarizing is ture.'
                )

    def __del__(self):
        if self.session is not None:
            self.session.close()

    @property
    def iter(self):
        return self._iter

    def get_batch_size(self):
        return int(self.cfg['batch_size']['batch_size'])

    def get_train_op(self):
        assert self.network is not None, 'Network not defined.'
        if self.cfg['optimization']['algorithm'] == 'AdaGrad':
            optimizer = tf.train.AdagradOptimizer(
                float(self.cfg['optimization']['learning_rate']),
                initial_accumulator_value=float(self.cfg['optimization']['delta']))
        return optimizer.apply_gradients(
            self.network.gradient(self.network.get_tensor_loss(),
                                  do_summarizing=self.do_summarizing)
        )

    def setup_network(self, network):
        assert issubclass(network.__class__, network_base.NetworkBase), \
            'network must be a subclass of NetworkBase class.'
        self.network = network
        graph = tf.Graph()
        with graph.as_default():
            network.setup()
            self.train_op = self.get_train_op()
            self.summaries = network.get_summaries()
            init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=graph, config=config)
        if self.do_summarizing:
            self.summary_writer = tf.summary.FileWriter(
                self.summary_path,
                self.session.graph
            )
        self.session.run(init)

    def train(self, x, y):
        if self.do_summarizing:
            _, summary_result, loss = self.session.run(
                [self.train_op, self.summaries, self.network.get_tensor_loss()],
                feed_dict={self.network.get_placeholder_x(): x,
                           self.network.get_placeholder_y(): y,
                           self.network.get_placeholder_keep_prob():
                               float(self.cfg['dropout']['probability'])}
            )
            self.summary_writer.add_summary(summary_result, self._iter)
            self.summary_writer.flush()
        else:
            _, loss = self.session.run(
                [self.train_op, self.network.get_tensor_loss()],
                feed_dict={self.network.get_placeholder_x(): x,
                           self.network.get_placeholder_y(): y,
                           self.network.get_placeholder_keep_prob():
                                float(self.cfg['dropout']['probability'])}
            )
        self._iter += 1
        return loss

    def test(self, x):
        return self.session.run(
            self.network.get_tensor_prediction(),
            feed_dict={self.network.get_placeholder_x(): x,
                       self.network.get_placeholder_keep_prob():
                           float(1.0)}
        )

    def get_max_epoch(self):
        return int(self.cfg['optimization']['epoch'])

    def save_model(self, path, save_type='tensorflow_save'):
        assert self.session is not None, ('Execute function self.setup_network '
                                         'before saving.')
        file_manager.create_if_not_exist(path)
        if save_type == 'tensorflow_save':
            with self.session.graph.as_default():
                saver = tf.train.Saver()
                saver.save(
                    self.session,
                    os.path.join(path, 'model.ckpt'),
                    self._iter
                )
        if save_type == 'npy_save':
            self.network.save_network_to_npy(
                os.path.join(path,
                             'model_{0}.npy'.format(self._iter)),
                self.session
            )

    def load_model(self, path, iter=None, load_type='tensorflow_save'):
        if iter is None:
            if load_type == 'tensorflow_save':
                last_modified_model_name = file_manager.last_modified(
                    os.path.join(path,
                                 'model.ckpt-*.data*')
                )
                iter = re.findall(
                    '\d+',
                    os.path.basename(last_modified_model_name)
                )[0]
                with self.session.graph.as_default():
                    saver = tf.train.Saver()
                    saver.restore(
                        self.session,
                        os.path.join(path,
                                     'model.ckpt-{0}'.format(iter))
                    )

            elif load_type == 'npy_save':
                last_modified_model_name = file_manager.last_modified(
                    os.path.join(path,
                                 'model_*.npy')
                )
                self.network.restore_network_from_npy(
                    last_modified_model_name,
                    self.session
                )
                iter = re.findall(
                    '\d+',
                    os.path.basename(last_modified_model_name)
                )[0]
            else:
                raise ValueError('load_type is illegal.')
        else:
            if load_type == 'tensorflow_save':
                with self.session.graph.as_default():
                    saver = tf.train.Saver()
                    try:
                        saver.restore(
                            self.session,
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
                    self.network.restore_network_from_npy(
                        os.path.join(
                            path,
                            'model_{0}.npy'.format(int(iter))
                        ),
                        self.session
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
        cfg_tools.write_cfg_file(path, self.cfg)
