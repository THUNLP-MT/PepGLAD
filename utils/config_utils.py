#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List


def format_args(args: List[str]):
    clean_args = []
    for arg in args:
        if not (arg.startswith('-') or arg.startswith('--')): # value
            clean_args.append(arg)
        else:
            arg = arg.lstrip('-').lstrip('-')
            clean_args.extend(arg.split('='))
    return clean_args


def get_parent_dict(config: dict, key: str):
    key_each_depth = key.split('.')
    for k in key_each_depth[:-1]:
        if k not in config:
            raise KeyError(f'Path key {key} not in the dict')
        config = config[k]
    return config, key_each_depth[-1] # last key


def overwrite_values(config, args):
    args = format_args(args)
    keys, values = args[0::2], args[1::2]
    for key, value in zip(keys, values):
        parent, last_key = get_parent_dict(config, key)
        ori_value = parent[last_key]
        parent[last_key] = type(ori_value)(value)
    return config
