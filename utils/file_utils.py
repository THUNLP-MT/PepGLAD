#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from os.path import basename, splitext


def get_filename(path):
    return basename(splitext(path)[0])


def cnt_num_files(directory, recursive=False):
    cnt = 0
    for sub in os.listdir(directory):
        sub = os.path.join(directory, sub)
        if os.path.isfile(sub):
            cnt += 1
        elif os.path.isdir(sub) and recursive:
            cnt += cnt_num_files(sub, recursive=True)
    return cnt