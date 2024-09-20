#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn_utils import graph_to_batch
from data.format import VOCAB

from .sidechain import SideChainBuilder, ChiAngles
from .constants import AA20


class SideChainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sidechain_builder = SideChainBuilder()
        self.chi_angle_calc = ChiAngles()

        aa_index_inverse_mapping = torch.tensor([VOCAB.symbol_to_idx(a) for a in AA20], dtype=torch.long)
        aa_index_mapping = torch.ones(aa_index_inverse_mapping.max() + 1, dtype=torch.long) * 20 # set 20 to unk (0~19 are natural amino acids)
        aa_index_mapping[aa_index_inverse_mapping] = torch.arange(20)
        self.register_buffer('aa_index_mapping', aa_index_mapping)

    def forward(self, X, S, batch_ids, optimize=True):
        '''
        X: [N, 14, 3], predicted all-atom coordinates (obviously with a lot of invalidities)
        S: [N], predicted sequence
        '''
        # do sequence index mapping from our vocabulary to the sidechain builder native indexes
        S = self.aa_index_mapping[S]

        # to batch-form representations
        X, mask = graph_to_batch(X, batch_ids, mask_is_pad=False)
        S, _ = graph_to_batch(S, batch_ids)
        C = mask.long()

        # rectify sidechains
        chi, _ = self.chi_angle_calc(X, C, S)
        ori_X = X.clone()
        if optimize:  # optimize chi so that the resulted atoms have similar positions with the predicted ones
            with torch.enable_grad():
                chi = chi.clone()
                chi.requires_grad = True
                delta, lr, step, last_mse = 1e-4, 1, 0, 100
                optimizer = torch.optim.Adam([chi], lr=lr)
                while True:
                    X, mask_X = self.sidechain_builder(ori_X[:, :, :4], C, S, chi)
                    mask_X = mask_X.squeeze(-1) # [bs, L, 14]
                    X, mask_X = X[:, :, 4:], mask_X[:, :, 4:].bool()
                    mse = F.mse_loss(X[mask_X], ori_X[:, :, 4:][mask_X]) # only on sidechain
                    if torch.abs(mse - last_mse) < delta:
                        break
                    mse.backward()
                    # chi.data = chi.data - lr * chi.grad.data
                    # chi.grad.zero_()
                    optimizer.step()
                    optimizer.zero_grad()
                    last_mse = mse.detach()
                    step += 1
                chi = chi.detach()
                # print(f'optimized {step} steps, mse {last_mse}')
        
        X, _ = self.sidechain_builder(ori_X[:, :, :4], C, S, chi)

        # get back to our graph representations
        return  X[mask]