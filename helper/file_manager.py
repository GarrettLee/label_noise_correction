import os
import glob
__author__ = 'garrett_local'


def last_modified(path_to_match):
    """
    Get the last modified file among those matching with a given wildcards.
    :param path_to_match: a wildcards indicating what files to find.
    :return: the last modified file among those matching with a given wildcards.
    """
    matched = glob.glob(path_to_match)
    matched.sort(key=lambda f: os.stat(f).st_mtime)
    newest = matched[-1]
    return newest


def create_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)