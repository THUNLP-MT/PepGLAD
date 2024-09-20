#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from utils.nn_utils import graph_to_batch

from .backbone import FrameBuilder


class BackboneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone_builder = FrameBuilder()

    def forward(self, X, batch_ids):
        '''
        X: [N, 14, 3], predicted all-atom coordinates (obviously with a lot of invalidities)
            assume the first 4 are N, CA, C, O
        S: [N], predicted sequence
        '''

        # to batch-form representations
        X, mask = graph_to_batch(X, batch_ids, mask_is_pad=False)
        C = mask.long()

        # rectify backbones
        R, t, q = self.backbone_builder.inverse(X, C)
        X_bb = self.backbone_builder(R, t, C) # [bs, L, 4, 3]
        X = torch.cat([X_bb, X[:, :, 4:]], dim=-2) # [bs, L, 14, 3]
        
        # get back to our graph representations
        return  X[mask]