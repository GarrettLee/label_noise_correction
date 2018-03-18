import numpy as np

__author__ = 'garrett_local'


def make_one_hot(labels, dim):
    """
    Create one-hot label with scalar labels.
    :param labels: a numpy ndarray in shape of [batch_size].
    :param dim: an int. dimension of output one-hot vector.
    :return: one-hot vectors in shape of [batch_size, dim]
    """
    return (np.arange(dim) == labels[:,None]).astype(np.integer)