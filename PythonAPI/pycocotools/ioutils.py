#!/usr/bin/python

__author__ = 'tillvolkmann'

import os
from shutil import copyfile
from sys import exit, exc_info


def list_files_in_dir(root_path, extension=None, sub_dirs=True):
    """
    List all files in root directory.

    :param root_path: path to directory to search; if root_path is a file, the path to the file is returned without exception
    :param extension: keep only files with extensions element of this str or list
    :param sub_dirs: include sub directories in search
    :return: list of files in directory
    """
    root_path = os.path.expanduser(root_path)
    assert os.path.exists(root_path), f"The source directory does not exist: {root_path}"
    assert os.path.isdir(root_path) or os.path.isfile(root_path), "root_path must either be directory or file"

    # get list of files in the directory
    if os.path.isdir(root_path):
        if sub_dirs:
            file_paths = []
            for dir_path, dir_names, file_names in os.walk(root_path):
                for file_name in file_names:
                    file_path = os.path.join(dir_path, file_name)
                    file_paths.append(file_path)
        else:
            file_paths = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f))]
    else:  # or pass on the single file name
        file_paths = root_path

    # filter paths to keep only files with specified extension(s)
    if extension is not None:
        assert type(extension) in [list, str], "Extension must be either string or list"
        # convert extension string to list if necessary
        if type(extension) is not list:
            extension = [extension]
        extension = [e.lower() for e in extension]
        file_paths = [f for f in file_paths if os.path.splitext(f)[1][1:].lower() in extension]

    return file_paths


def copy_file(image_path, target_path):
    try:
        copyfile(image_path, target_path)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", exc_info())
        exit(1)


if __name__ == '__main__':
    pass