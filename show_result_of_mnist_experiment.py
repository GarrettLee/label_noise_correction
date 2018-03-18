import numpy as np
import matplotlib.pyplot as plt

__author__ = 'garrett_local'


def parse_result(result):
    x = []
    y = []
    for idx, acc in enumerate(result):
        if acc != -1:  # acc equal to -1, meaning singular matrix problem exist.
            x.append(idx / 10.)
            y.append(acc)
    return x,y


if __name__ == '__main__':
    try:
        x_cross_entropy, y_cross_entropy = \
            parse_result(np.load('./result/mnist/cross_entropy.npy'))
        x_forward, y_forward = \
            parse_result(np.load('./result/mnist/forward.npy'))
        x_forward_t, y_forward_t = \
            parse_result(np.load('./result/mnist/forward_t.npy'))
        x_backward, y_backward = \
            parse_result(np.load('./result/mnist/backward.npy'))
        x_backward_t, y_backward_t = \
            parse_result(np.load('./result/mnist/backward_t.npy'))
    except IOError, e:
        print e.message
        print ('No experiment result found, first run bash script '
              'run_experiment_mnist.')

    handler_cross_entropy, = plt.plot(x_cross_entropy, y_cross_entropy, 'yx-')
    handler_backward, = plt.plot(x_backward, y_backward, 'rx-')
    handler_backward_t, = plt.plot(x_backward_t, y_backward_t, 'rx--')
    handler_forward, = plt.plot(x_forward, y_forward, 'bx-')
    handler_forward_t, = plt.plot(x_forward_t, y_forward_t, 'bx--')
    plt.legend(
        [handler_cross_entropy,
         handler_backward,
         handler_backward_t,
         handler_forward,
         handler_forward_t],
        ['Cross Entropy',
         'Backward',
         'Backward T',
         'Forward',
         'Forward T'],
        loc='lower left')
    plt.show()
