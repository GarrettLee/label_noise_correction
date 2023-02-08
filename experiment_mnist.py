#!/usr/bin/python2
from typing import Tuple
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import trainer.trainer_base as trainer_base
import network.fc_mnist_network as fc_mnist_network
import helper.ml_helper as ml_helper
import helper.noisy_labels_creator as noisy_labels_creator
import helper.math_helpers as math_helpers

__author__ = 'garrett_local'

cfg_path = './cfg/experiments/mnist/mnist'


def get_mnist_batch(
    dataset_predictors: np.ndarray,
    dataset_labels: np.ndarray,
    batch_size: int,
    start: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples a batch of data from both the predictors and labels.
    If `start` is given, will sample rows subsequently following the row indexed at `start.
    Otherwise, samples randomly.
    """
    assert dataset_predictors.shape[0] == dataset_labels.shape[0], AssertionError("Predictors and labels must have same number of samples")

    if start:
        feature_batch, label_batch = dataset_predictors[start:start+batch_size, :], dataset_labels[start:start+batch_size]
    else:
        random_index = np.random.choice(dataset_predictors.shape[0], batch_size, replace=False)
        feature_batch, label_batch = dataset_predictors[random_index, :], dataset_labels[random_index]

    
    return (
        np.reshape(feature_batch, (feature_batch.shape[0], feature_batch.shape[1]*feature_batch.shape[2])),
        label_batch
    )


def initialize_model_and_train(
    n,
    t_matrix,
    model_path,
    summary_path,
    loss_type = 'cross_entropy'
)-> Tuple[trainer_base.TrainerBase, np.ndarray, np.ndarray]:

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    trainer = trainer_base.TrainerBase(
        cfg_path=cfg_path,
        do_summarizing=False,
        summary_path=summary_path
    )
    trainer.setup_network(fc_mnist_network.FcMnistNetwork(
        do_summarizing=False,
        loss_type=loss_type,
        transition_mat=t_matrix
    ))
    if trainer.load_model(
            model_path,
            iter=17618
    ) is None:
        epochs_completed: int = 0
        while epochs_completed <= trainer.get_max_epoch():
            train_batch: Tuple[np.ndarray, np.ndarray] = get_mnist_batch(
                dataset_predictors=x_train,
                dataset_labels=y_train,
                batch_size=trainer.get_batch_size()
            )
            noisy_labels = noisy_labels_creator.create_mnist_noisy_labels(
                n,
                ml_helper.make_one_hot(train_batch[1], 10)
            )

            loss = trainer.train(
                (train_batch[0], noisy_labels)
            )
            if trainer.iter % 100 == 0:
                print('iter: {0}, loss: {1}'.format(trainer.iter, loss))
            epochs_completed += 1

    return trainer, x_test, y_test


def test_model_accuracy(trainer, x_test, y_test)-> float:
    tested = 0
    pred = []
    while tested < x_test.shape[0]:
        left = x_test.shape[0] - tested
        if left >= trainer.get_batch_size():
            test_batch: Tuple[np.ndarray, np.ndarray] = get_mnist_batch(
                dataset_predictors=x_test,
                dataset_labels=y_test,
                batch_size=trainer.get_batch_size(),
                start=tested
            )
        else:
            test_batch: Tuple[np.ndarray, np.ndarray] = get_mnist_batch(
                dataset_predictors=x_test,
                dataset_labels=y_test,
                batch_size=left,
                start=tested
            )
        pred_one_hot = trainer.test((test_batch[0], ))
        pred.extend(np.argmax(pred_one_hot, axis=1))
        tested += trainer.get_batch_size()
    np.array(pred)
    acc = (np.sum(np.array(pred) == y_test) /
           float(x_test.shape[0]))

    return acc


def backward_experiment(n, testing_index, t_matrix, experiment_name):

    # Singular transition matrix.
    if np.linalg.det(noisy_labels_creator.create_mnist_transition_matrix(n)) == 0:
        return -1

    cross_entropy_model_path = \
        ('./model/network/experiment_mnist/'
         '{}_n_{:.1f}_test_{:.0f}'.format(experiment_name, n, testing_index))
    summary_path = \
        ('./summaries/network/experiment_mnist/'
         '{}_{:.1f}_test_{:.0f}'.format(experiment_name, n, testing_index))

    trainer, x_test, y_test = initialize_model_and_train(
        n=n,
        t_matrix=t_matrix,
        model_path=cross_entropy_model_path,
        summary_path=summary_path,
        loss_type='backward',
    )

    acc = test_model_accuracy(
        trainer=trainer,
        x_test=x_test,
        y_test=y_test
    )
    print('Accuracy: {0}'.format(acc))
    trainer.save_model(cross_entropy_model_path)
    return acc


def forward_experiment(n, testing_index, t_matrix, experiment_name):
    cross_entropy_model_path = \
        ('./model/network/experiment_mnist/'
         '{}_n_{:.1f}_test_{:.0f}'.format(experiment_name, n, testing_index))
    summary_path = \
        ('./summaries/network/experiment_mnist/'
         '{}_{:.1f}_test_{:.0f}'.format(experiment_name, n, testing_index))

    trainer, x_test, y_test = initialize_model_and_train(
        n=n,
        t_matrix=t_matrix,
        model_path=cross_entropy_model_path,
        summary_path=summary_path,
        loss_type='forward',
    )

    acc = test_model_accuracy(
        trainer=trainer,
        x_test=x_test,
        y_test=y_test
    )
    print('Accuracy: {0}'.format(acc))
    trainer.save_model(cross_entropy_model_path)
    return acc


def cross_entropy_experiment(n, testing_index):
    cross_entropy_model_path = \
        ('./model/network/experiment_mnist/'
         'cross_entropy_n_{:.1f}_test_{:.0f}'.format(n, testing_index))
    summary_path = \
        ('./summaries/network/experiment_mnist/'
         'cross_entropy_{:.1f}_test_{:.0f}'.format(n, testing_index))

    trainer, x_test, y_test = initialize_model_and_train(
        n=n,
        t_matrix=None,
        model_path=cross_entropy_model_path,
        summary_path=summary_path,
    )

    acc = test_model_accuracy(
        trainer=trainer,
        x_test=x_test,
        y_test=y_test
    )
    print('Accuracy: {0}'.format(acc))
    trainer.save_model(cross_entropy_model_path)
    return acc


def estimate_t(n, testing_index, percentile=97):
    cross_entropy_model_path = \
        ('./model/network/experiment_mnist/'
         'estimator_n_{:.1f}_test_{:.0f}'.format(n, testing_index))
    summary_path = \
        ('./summaries/network/experiment_mnist/'
         'estimator_{:.1f}_test_{:.0f}'.format(n, testing_index))

    #mnist = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    trainer, x_test, y_test = initialize_model_and_train(
        n=n,
        t_matrix=None,
        model_path=cross_entropy_model_path,
        summary_path=summary_path,
    )
    trainer.save_model(cross_entropy_model_path)
    probabilities = None

    tested = 0
    while tested < x_train.shape[0]:
        left = x_train.shape[0] - tested
        if left >= trainer.get_batch_size():
            test_batch = get_mnist_batch(
                dataset_predictors=x_train,
                dataset_labels=y_train,
                batch_size=trainer.get_batch_size(),
                start=tested
            )
        else:
            test_batch = get_mnist_batch(
                dataset_predictors=x_train,
                dataset_labels=y_train,
                batch_size=left,
                start=tested
            )
        pred_one_hot = trainer.test((test_batch[0], ))
        if probabilities is None:
            probabilities = pred_one_hot
        else:
            probabilities = np.vstack((probabilities, pred_one_hot))
        tested += trainer.get_batch_size()

    print(pred_one_hot)

    tested = 0
    while tested < x_test.shape[0]:
        left = x_test.shape[0] - tested
        if left >= trainer.get_batch_size():
            test_batch = get_mnist_batch(
                dataset_predictors=x_test,
                dataset_labels=y_test,
                batch_size=trainer.get_batch_size(),
                start=tested
            )
        else:
            test_batch = get_mnist_batch(
                dataset_predictors=x_test,
                dataset_labels=y_test,
                batch_size=left,
                start=tested
            )
        pred_one_hot = trainer.test((test_batch[0], ))
        if probabilities is None:
            probabilities = pred_one_hot
        else:
            probabilities = np.vstack((probabilities, pred_one_hot))
        tested += trainer.get_batch_size()


    t_matrix_estimated = np.eye(10)
    for i in range(10):
        if percentile == 1:
            x_overbar_idx = np.argmax(probabilities[:, i])
        else:
            x_overbar_idx = math_helpers.arg_percentile(
                probabilities[:, i],
                percentile
            )
        for j in range(10):
            t_matrix_estimated[i, j] = probabilities[x_overbar_idx, j]
    return t_matrix_estimated


def mnist_experiment_backward():
    exp_data = []
    for n in np.arange(0, 1, 0.1):
        exp_data_n = []
        for testing_index in range(5):
            exp_data_n.append(backward_experiment(
                n,
                testing_index,
                noisy_labels_creator.create_mnist_transition_matrix(n),
                'backward'
            ))
        exp_data.append(exp_data_n)
    result = np.mean(exp_data, axis=1)
    print(result)
    plt.plot(result)
    np.save('./result/mnist/backward.npy', result)
    #$ plt.show(1)


def mnist_experiment_forward():
    exp_data = []
    for n in np.arange(0, 1, 0.1):
        exp_data_n = []
        for testing_index in range(5):
            exp_data_n.append(forward_experiment(
                n,
                testing_index,
                noisy_labels_creator.create_mnist_transition_matrix(n),
                'forward'
            ))
        exp_data.append(exp_data_n)
    result = np.mean(exp_data, axis=1)
    print(result)
    plt.plot(result)
    np.save('./result/mnist/forward.npy', result)
    # plt.show(1)


def mnist_experiment_cross_entropy():
    exp_data = []
    for n in np.arange(0, 1, 0.1):
        exp_data_n = []
        for testing_index in range(5):
            exp_data_n.append(cross_entropy_experiment(n, testing_index))
        exp_data.append(exp_data_n)
    result = np.mean(exp_data, axis=1)
    print(result)
    plt.plot(result)
    np.save('./result/mnist/cross_entropy.npy', result)
    # plt.show(1)


def mnist_experiment_backward_t():
    exp_data = []
    for n in np.arange(0, 1, 0.1):
        exp_data_n = []
        for testing_index in range(5):
            t_matrix = estimate_t(n, testing_index)
            print(f"estimated t_matrix at n = {n}: ")
            print(t_matrix)
            exp_data_n.append(backward_experiment(
                n,
                testing_index,
                t_matrix,
                'backward_t'
            ))
        exp_data.append(exp_data_n)
    result = np.mean(exp_data, axis=1)
    print(result)
    plt.plot(result)
    np.save('./result/mnist/backward_t.npy', result)

    # plt.show(1)


def mnist_experiment_forward_t():
    exp_data = []
    for n in np.arange(0, 1, 0.1):
        exp_data_n = []
        for testing_index in range(5):
            t_matrix = estimate_t(n, testing_index)
            print(f"estimated t_matrix at n = {n}: ")
            print(t_matrix)
            exp_data_n.append(forward_experiment(
                n,
                testing_index,
                t_matrix,
                'forward_t'
            ))
        exp_data.append(exp_data_n)
    result = np.mean(exp_data, axis=1)
    print(result)
    plt.plot(result)
    np.save('./result/mnist/forward_t.npy', result)
    # plt.show(1)

if __name__ == '__main__':
    result_path = './result/mnist/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if len(sys.argv) == 1:
        mnist_experiment_backward_t()
        mnist_experiment_forward_t()
        mnist_experiment_backward()
        mnist_experiment_forward()
        mnist_experiment_cross_entropy()
        plt.show(1)
    else:
        if sys.argv[1] == 'backward_t':
            mnist_experiment_backward_t()
            plt.show(1)
        elif sys.argv[1] == 'forward_t':
            mnist_experiment_forward_t()
            plt.show(1)
        elif sys.argv[1] == 'backward':
            mnist_experiment_backward()
            plt.show(1)
        elif sys.argv[1] == 'forward':
            mnist_experiment_forward()
            plt.show(1)
        elif sys.argv[1] == 'cross_entropy':
            mnist_experiment_cross_entropy()
            plt.show(1)
        else:
            raise ValueError('Invalid argument: {}'.format(sys.argv[1]))
