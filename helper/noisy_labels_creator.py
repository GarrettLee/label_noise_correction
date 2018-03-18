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
    samples_num = one_hot_label.shape[0]
    sample = (np.random.uniform(size=(
        samples_num,
        t_matrix.shape[0],
        t_matrix.shape[1]
    )) <= t_matrix)
    for idx, label in enumerate(one_hot_label):
        one_hot_label[idx] = np.matmul(one_hot_label[idx:idx+1], sample[idx])
    return one_hot_label


def create_mnist_noisy_labels(n, one_hot_label, with_eye=False):
    return create_mnist_noisy_labels_with_t(
        create_mnist_transition_matrix(n, with_eye=with_eye),
        one_hot_label
    )

