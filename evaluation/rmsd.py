#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np


# a: [N, 3], b: [N, 3]
def compute_rmsd(a, b, aligned=False):  # amino acids level rmsd
    dist = np.sum((a - b) ** 2, axis=-1)
    rmsd = np.sqrt(dist.sum() / a.shape[0])
    return float(rmsd)