import os
from glob import glob


def files_form_folder(folder, filename='*'):
    """
    Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards à la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.
    """

    if not os.path.exists(folder):
        raise FileNotFoundError(f"'{folder}' Directory not found!")
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"The path '{folder}' is not a directory!")
    filenames = sorted(glob(os.path.join(folder, filename)))
    return filenames
