#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from data.format import VOCAB
from utils.nn_utils import variadic_meshgrid
from utils import register as R
from utils.oom_decorator import oom_decorator

from .modules.am_egnn import AMEGNN
from .nn_utils import SeparatedAminoAcidFeature, ProteinFeature


@R.register('dyMEAN')
class dyMEAN(nn.Module):
    def __init__(
            self,
            embed_size,
            hidden_size,
            n_channel,
            num_classes=len(VOCAB),
            mask_id=VOCAB.get_mask_idx(),
            max_position=2048,
            CA_channel_idx=VOCAB.backbone_atoms.index('CA'),
            n_layers=3,
            iter_round=3,
            dropout=0.1,
            fix_atom_weights=False,
            relative_position=False,
            mode='codesign',  # fixbb, fixseq(structure prediction), codesign
            std=10.0
        ) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.ca_channel_idx = CA_channel_idx
        self.round = iter_round
        self.mode = mode
        self.std = std

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            max_position=max_position,
            relative_position=relative_position,
            fix_atom_weights=fix_atom_weights
        )
        self.protein_feature = ProteinFeature()
        
        self.memory_ffn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, embed_size)
        )
        if self.mode != 'fixseq':
            self.ffn_residue = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )
        else:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.gnn = AMEGNN(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False)
        
        # training related cache
        self.batch_constants = {}

    def init_mask(self, X, S, xmask, smask, batch_ids):
        X, S = X.clone(), S.clone() # [N, 14, 3], [N]
        n_channel, n_dim = X.shape[1:]
        if self.mode != 'fixseq':
            S[smask] = self.mask_id
        
        if self.mode != 'fixbb':
            receptor_centers = scatter_mean(X[~xmask][:, self.ca_channel_idx], batch_ids[~xmask], dim=0) # [bs, 3]
            ligand_ca = torch.randn_like(X[xmask][:, self.ca_channel_idx]) * self.std + receptor_centers[batch_ids[xmask]]  # [Nlig, 3]
            ligand_X = ligand_ca.unsqueeze(1).repeat(1, n_channel, 1)
            ligand_X = ligand_X + torch.randn_like(ligand_X) * self.std * 0.1 # smaller scale
            X[xmask] = ligand_X

        return X, S

    def message_passing(self, X, S, position_ids, ctx_edges, inter_edges, atom_weights, memory_H=None, smooth_prob=None, smooth_mask=None):
        # embeddings
        H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids, smooth_prob=smooth_prob, smooth_mask=smooth_mask)

        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)
        edges = torch.cat([ctx_edges, inter_edges], dim=1)

        H, pred_X = self.gnn(H_0, X, edges,
                             channel_attr=atom_embeddings,
                             channel_weights=atom_weights)


        pred_logits = None if self.mode == 'fixseq' else self.ffn_residue(H)

        return pred_logits, pred_X, H  # [N, num_classes], [N, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def prepare_inputs(self, X, S, xmask, smask, lengths):

        # batch ids
        batch_ids = torch.zeros_like(S)
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)

        # initialization
        X, S = self.init_mask(X, S, xmask, smask, batch_ids)
        aa_cnt = smask.sum()

        # edges
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)

        is_ctx = xmask[row] == xmask[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        special_mask = special_mask.repeat(aa_cnt, 1).bool()

        return X, S, aa_cnt, ctx_edges, inter_edges, special_mask, batch_ids

    def normalize(self, X, batch_ids, mask):
        centers = scatter_mean(X[~mask], batch_ids[~mask], dim=0) # [bs, 4, 3]
        centers = centers.mean(dim=1)[batch_ids].unsqueeze(1) # [N, 4, 3]
        X = (X - centers) / self.std
        return X, centers

    def _forward(self, X, S, xmask, smask, special_mask, position_ids, ctx_edges, inter_edges, atom_weights):

        # sequence and structure loss
        r_pred_S_logits, pred_S_dist = [], None
        memory_H = None
        # message passing
        for t in range(self.round):
            pred_S_logits, pred_X, H = self.message_passing(
                X, S, position_ids, ctx_edges, inter_edges,
                atom_weights, memory_H, pred_S_dist, smask)
            r_pred_S_logits.append(pred_S_logits)
            memory_H = H
            # 1. update X
            X = X.clone()
            X[xmask] = pred_X[xmask]

            if self.mode != 'fixseq':
                # 2. update S
                S = S.clone()
                if t == self.round - 1:
                    S[smask] = torch.argmax(pred_S_logits[smask].masked_fill(special_mask, float('-inf')), dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_S_logits[smask].masked_fill(special_mask, float('-inf')), dim=-1)

        if self.mode == 'fixseq':
            # predicted rmsd
            prmsd = self.prmsd_ffn(H[xmask]).squeeze()  # [N_ab]
        else:
            prmsd = None

        return H, S, r_pred_S_logits, pred_X, prmsd

    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, context_ratio=0):
        '''
        :param X: [N, 14, 3]
        :param S: [N]
        :param smask: [N]
        :param position_ids: [N], residue position ids
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training
        '''
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()
        xmask, smask = mask, mask
        
        # provide some ground truth for annealing sequence training
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)

        # prepare
        X, S, aa_cnt, ctx_edges, inter_edges, special_mask, batch_ids = self.prepare_inputs(X, S, xmask, smask, lengths)
        atom_weights = torch.logical_or(atom_mask, xmask.unsqueeze(1)).float() if self.mode != 'fixbb' else atom_mask.float()
        X, centers = self.normalize(X, batch_ids, xmask)
        true_X, _ = self.normalize(true_X, batch_ids, xmask)

        # get results
        H, pred_S, r_pred_S_logits, pred_X, prmsd = self._forward(
            X, S, xmask, smask, special_mask, position_ids, ctx_edges, inter_edges, atom_weights)
        # # unnormalize
        # pred_X = pred_X * self.std + centers

        # sequence negtive log likelihood
        snll = 0
        if self.mode != 'fixseq':
            for logits in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[smask].masked_fill(special_mask, float('-inf')), true_S[smask], reduction='sum') / (aa_cnt + 1e-10)
            snll = snll / self.round

        # coordination loss
        if self.mode != 'fixbb':
            segment_ids, gen_X, ref_X = torch.ones_like(pred_S[xmask], device=pred_X.device, dtype=torch.long), pred_X[xmask], true_X[xmask]
            # backbone bond lengths loss
            bb_bond_loss = F.l1_loss(
                self.protein_feature._cal_backbone_bond_lengths(gen_X, batch_ids[xmask], segment_ids, atom_mask[xmask]),
                self.protein_feature._cal_backbone_bond_lengths(ref_X, batch_ids[xmask], segment_ids, atom_mask[xmask])
            )
            # side-chain bond lengths loss
            sc_bond_loss = F.l1_loss(
                self.protein_feature._cal_sidechain_bond_lengths(true_S[xmask], gen_X, self.aa_feature, atom_mask[xmask]),
                self.protein_feature._cal_sidechain_bond_lengths(true_S[xmask], ref_X, self.aa_feature, atom_mask[xmask])
            )
            # mse
            xloss_mask = atom_mask.unsqueeze(-1).repeat(1, 1, 3) & mask.unsqueeze(-1).unsqueeze(-1) # [N, 14, 3]
            xloss = F.mse_loss(pred_X[xloss_mask], true_X[xloss_mask])
            # CA pair-wise distance
            dist_loss = F.l1_loss(
                torch.norm(pred_X[:, self.ca_channel_idx][inter_edges.T[0]] - pred_X[:, self.ca_channel_idx][inter_edges.T[1]], dim=-1),
                torch.norm(true_X[:, self.ca_channel_idx][inter_edges.T[0]] - true_X[:, self.ca_channel_idx][inter_edges.T[1]], dim=-1)
            )
            struct_loss = bb_bond_loss + sc_bond_loss + xloss + dist_loss
        else:
            struct_loss, bb_bond_loss, sc_bond_loss, xloss, dist_loss = 0, 0, 0, 0, 0

        if self.mode != 'fixbb':
            # predicted rmsd
            prmsd_loss = 0 # TODO: residue-wise rmsd
            pdev_loss = prmsd_loss# + prmsd_i_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss, 5 for similar scale
        loss = snll + 5 * struct_loss + (0 if pdev_loss is None else pdev_loss)#  + 0 * ed_loss

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / (aa_hit.shape[0] + 1e-10)

        return loss, (snll, aar), (struct_loss, (bb_bond_loss, sc_bond_loss, xloss, dist_loss)), (pdev_loss, prmsd_loss)# , (ed_loss, r_ed_losses)

    def sample(self, X, S, mask, position_ids, lengths, atom_mask, greedy=False):
        gen_X, gen_S = X.clone(), S.clone()
        xmask, smask = mask, mask
        
        # prepare
        X, S, aa_cnt, ctx_edges, inter_edges, special_mask, batch_ids = self.prepare_inputs(X, S, xmask, smask, lengths)
        atom_weights = torch.logical_or(atom_mask, xmask.unsqueeze(1)).float() if self.mode != 'fixbb' else atom_weights.float()
        X, centers = self.normalize(X, batch_ids, xmask)

        # get results
        H, pred_S, r_pred_S_logits, pred_X, prmsd = self._forward(X, S, xmask, smask, special_mask, position_ids, ctx_edges, inter_edges, atom_weights)
        # unnormalize
        pred_X = pred_X * self.std + centers


        if self.mode != 'fixseq':

            logits = r_pred_S_logits[-1][smask]
            logits = logits.masked_fill(special_mask, float('-inf'))  # mask special tokens

            if greedy:
                gen_S[smask] = torch.argmax(logits, dim=-1)  # [n]
            else:
                prob = F.softmax(logits, dim=-1)
                gen_S[smask] = torch.multinomial(prob, num_samples=1).squeeze()
            snll_all = F.cross_entropy(logits, gen_S[smask], reduction='none')
        else:
            snll_all = torch.zeros_like(gen_S[smask]).float()

        gen_X[xmask] = pred_X[xmask]

        batch_X, batch_S, batch_ppls = [], [], []
        for i, l in enumerate(lengths):
            cur_mask = mask & (batch_ids == i)
            batch_X.append(gen_X[cur_mask].tolist())
            batch_S.append(''.join([VOCAB.idx_to_symbol(s) for s in gen_S[cur_mask]]))
            batch_ppls.append(
                torch.exp(snll_all[cur_mask[mask]].sum() / cur_mask.sum()).item()
            )
        return batch_X, batch_S, batch_ppls
