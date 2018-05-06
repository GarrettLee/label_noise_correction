import numpy as np
__author__ = 'garrett_local'


def create_mnist_transition_matrix(n, with_eye=False):
    t_matrix = np.array(
        [[1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
         [0,   1,   0,   0,   0,   0,   0,   0,   0,   0],
         [0,   0,   1-n, 0,   0,   0,   0,   n,   0,   0],
         [0,   0,   0,   1-n, 0,   0,   0,   0,   n,   0],
         [0,   0,   0,   0,   1,   0,   0,   0,   0,   0],
         [0,   0,   0,   0,   0,   1-n, n,   0,   0,   0],
         [0,   0,   0,   0,   0,   n,   1-n, 0,   0,   0],
         [0,   n,   0,   0,   0,   0,   0,   1-n, 0,   0],
         [0,   0,   0,   0,   0,   0,   0,   0,   1,   0],
         [0,   0,   0,   0,   0,   0,   0,   0,   0,   1]]
    )
    if with_eye:
        t_matrix += np.eye(10)
    return t_matrix


def create_mnist_noisy_labels_with_t(t_matrix, one_hot_label):

    # sampling
    pdf = np.matmul(one_hot_label, t_matrix)
    cdf = np.matmul(pdf, np.tri(pdf.shape[1]).transpose())
    sample = np.random.uniform(size=one_hot_label.shape[0])
    t = np.eye(pdf.shape[1]) - np.eye(pdf.shape[1], k=1)
    ret = np.matmul((cdf - sample.reshape([-1, 1])) > 0, t)
    return ret


def create_mnist_noisy_labels(n, one_hot_label, with_eye=False):
    return create_mnist_noisy_labels_with_t(
        create_mnist_transition_matrix(n, with_eye=with_eye),
        one_hot_label
    )

