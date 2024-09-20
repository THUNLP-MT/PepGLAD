#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats.contingency import association

from evaluation.seq_metric import align_sequences


def seq_diversity(seqs: List[str], th: float=0.4) -> float:
    '''
        th: sequence distance 
    '''
    dists = []
    for i, seq1 in enumerate(seqs):
        dists.append([])
        for j, seq2 in enumerate(seqs):
            _, sim = align_sequences(seq1, seq2)
            dists[i].append(1 - sim)
    dists = np.array(dists)
    Z = linkage(squareform(dists), 'single')
    cluster = fcluster(Z, t=th, criterion='distance')
    return len(np.unique(cluster)) / len(seqs), cluster


def struct_diversity(structs: np.ndarray, th: float=4.0) -> float:
    '''
    structs: N*L*3, alpha carbon coordinates
    th: threshold for clustering (distance < th)
    '''
    ca_dists = np.sum((structs[:, None] - structs[None, :]) ** 2, axis=-1) # [N, N, L]
    rmsd = np.sqrt(np.mean(ca_dists, axis=-1))
    Z = linkage(squareform(rmsd), 'single') # since the distances might not be euclidean distances (e.g. rmsd)
    cluster = fcluster(Z, t=th, criterion='distance')
    return len(np.unique(cluster)) / structs.shape[0], cluster


def diversity(seqs: List[str], structs: np.ndarray):
    seq_div, seq_clu = seq_diversity(seqs)
    if structs is None:
        return seq_div, None, seq_div, None
    struct_div, struct_clu = struct_diversity(structs)
    co_div = np.sqrt(seq_div * struct_div)

    n_seq_clu, n_struct_clu = np.max(seq_clu), np.max(struct_clu) # clusters start from 1
    if n_seq_clu == 1 or n_struct_clu == 1:
        consistency = 1.0 if n_seq_clu == n_struct_clu else 0.0
    else:
        table = [[0 for _ in range(n_struct_clu)] for _ in range(n_seq_clu)]
        for seq_c, struct_c in zip(seq_clu, struct_clu):
            table[seq_c - 1][struct_c - 1] += 1
        consistency = association(np.array(table), method='cramer')

    return seq_div, struct_div, co_div, consistency


if __name__ == '__main__':
    N, L = 100, 10
    a = np.random.randn(N, L, 3)
    print(struct_diversity(a))
    from utils.const import aas
    aas = [tup[0] for tup in aas]
    seqs = np.random.randint(0, len(aas), (N, L))
    seqs = [''.join([aas[i] for i in idx]) for idx in seqs]
    print(seq_diversity(seqs, 0.4))