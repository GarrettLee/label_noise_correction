import numpy as np

__author__ = 'garrett_local'


def arg_percentile(np_array, percent):
    """
    Return the index of a specified percentile. The index would be index of
    percentile in a flattened array.
    :param np_array: an ndarray.
    :param percent: a float. It is the number of percents of the percentile.
    :return: The index of the target percentile in a flattened array.
    """
    assert percent >= 0 and percent <= 100, \
        "Argument percent is invalid."
    percentile = np.percentile(np_array, percent)
    return np.argmin(np.abs(np_array - percentile))