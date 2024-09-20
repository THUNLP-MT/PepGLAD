#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np


class ClusterResampler:
    def __init__(self, cluster_path: str) -> None:
        idx2prob = []
        with open(cluster_path, 'r') as fin:
            for line in fin:
                cluster_n_member = int(line.strip().split('\t')[-1])
                idx2prob.append(1 / cluster_n_member)
        total = sum(idx2prob)
        idx2prob = [p / total for p in idx2prob]
        self.idx2prob = np.array(idx2prob)

    def __call__(self, n_sample:int, replace: bool=False):
        idxs = np.random.choice(len(self.idx2prob), size=n_sample, replace=replace, p=self.idx2prob)
        return idxs