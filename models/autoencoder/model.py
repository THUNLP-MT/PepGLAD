#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from data.format import VOCAB
from utils import register as R
from utils.oom_decorator import oom_decorator
from utils.const import aas
from utils.nn_utils import variadic_meshgrid

from .sidechain.api import SideChainModel
from .backbone.api import BackboneModel

from ..dyMEAN.modules.am_egnn import AMEGNN # adaptive-multichannel egnn
from ..dyMEAN.nn_utils import SeparatedAminoAcidFeature, ProteinFeature


def create_encoder(
    name,
    atom_embed_size,
    embed_size,
    hidden_size,
    n_channel,
    n_layers,
    dropout,
    n_rbf,
    cutoff
):
    if name == 'dyMEAN':
        encoder = AMEGNN(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
    else:
        raise NotImplementedError(f'Encoder {encoder} not implemented')

    return encoder



@R.register('AutoEncoder')
class AutoEncoder(nn.Module):
    def __init__(
            self,
            embed_size,
            hidden_size,
            latent_size,
            n_channel,
            latent_n_channel=1,
            mask_id=VOCAB.get_mask_idx(),
            latent_id=VOCAB.symbol_to_idx(VOCAB.LAT),
            max_position=2048,
            relative_position=False,
            CA_channel_idx=VOCAB.backbone_atoms.index('CA'),
            n_layers=3,
            dropout=0.1,
            mask_ratio=0.0,
            fix_alpha_carbon=False,
            h_kl_weight=0.1,
            z_kl_weight=0.5,
            coord_loss_weights={
                'Xloss': 1.0,
                'ca_Xloss': 0.0,
                'bb_bond_lengths_loss': 1.0,
                'sc_bond_lengths_loss': 1.0,
                'bb_dihedral_angles_loss': 0.0, # this significantly poison the training
                'sc_chi_angles_loss': 0.5
            },
            coord_loss_ratio=0.5,  # (1 - r)*seq + r * coord
            coord_prior_var=1.0,   # sigma^2
            anchor_at_ca=False,
            share_decoder=False,
            n_rbf=0,
            cutoff=0,
            encoder='dyMEAN',
            mode='codesign' # codesign, fixbb (inverse folding), fixseq (structure prediction)
        ) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.latent_id = latent_id
        self.ca_channel_idx = CA_channel_idx
        self.n_channel = n_channel
        self.mask_ratio = mask_ratio
        self.fix_alpha_carbon = fix_alpha_carbon
        self.h_kl_weight = h_kl_weight
        self.z_kl_weight = z_kl_weight
        self.coord_loss_weights = coord_loss_weights
        self.coord_loss_ratio = coord_loss_ratio
        self.mode = mode
        self.latent_size = 0 if self.mode == 'fixseq' else latent_size
        self.latent_n_channel = 0 if self.mode == 'fixbb' else latent_n_channel
        self.anchor_at_ca = anchor_at_ca
        self.coord_prior_var = coord_prior_var

        if self.fix_alpha_carbon: assert self.latent_n_channel == 1, f'Specifying fix alpha carbon (use Ca as the latent coordinate) but number of latent channels is not 1'
        if self.anchor_at_ca: assert self.latent_n_channel == 1, f'Specifying anchor_at_ca as True but number of latent channels is not 1'
        if self.mode == 'fixseq': assert self.coord_loss_ratio == 1.0, f'Specifying fixseq mode but coordination loss ratio is not 1.0: {self.coord_loss_ratio}'
        if self.mode == 'fixbb': assert self.coord_loss_ratio == 0.0, f'Specifying fixbb mode but coordination loss ratio is not 0.0: {self.coord_loss_ratio}'
        
        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            max_position=max_position,
            relative_position=relative_position,
            fix_atom_weights=True
        )
        self.protein_feature = ProteinFeature()
        
        self.encoder = create_encoder(
            name = encoder,
            atom_embed_size = atom_embed_size,
            embed_size = embed_size,
            hidden_size = hidden_size,
            n_channel = n_channel,
            n_layers = n_layers,
            dropout = dropout,
            n_rbf = n_rbf,
            cutoff = cutoff
        )
        
        if self.mode != 'fixbb':
            self.sidechain_decoder = create_encoder(
                name = encoder,
                atom_embed_size = atom_embed_size,
                embed_size = embed_size,
                hidden_size = hidden_size,
                n_channel = n_channel,
                n_layers = n_layers,
                dropout = dropout,
                n_rbf = n_rbf,
                cutoff = cutoff
            )
            self.backbone_model = BackboneModel()
            self.sidechain_model = SideChainModel()
            self.W_Z_log_var = nn.Linear(hidden_size, latent_n_channel * 3)
        
        if self.mode != 'fixseq':
            self.W_mean = nn.Linear(hidden_size, latent_size)
            self.W_log_var = nn.Linear(hidden_size, latent_size)
            # self.hidden2latent = nn.Linear(hidden_size, latent_size)
            self.latent2hidden = nn.Linear(latent_size, hidden_size)
            self.merge_S_H = nn.Linear(hidden_size * 2, hidden_size)

            if share_decoder:
                self.seq_decoder = self.sidechain_decoder
            else:
                self.seq_decoder = create_encoder(
                    name = encoder,
                    atom_embed_size = atom_embed_size,
                    embed_size = embed_size,
                    hidden_size = hidden_size,
                    n_channel = n_channel,
                    n_layers = n_layers,
                    dropout = dropout,
                    n_rbf = n_rbf,
                    cutoff = cutoff
                )
        
        # residue type index mapping, from original index to 0~20, 0 is unk
        self.unk_idx = 0
        self.s_map = [0 for _ in range(len(VOCAB))]
        self.s_remap = [0 for _ in range(len(aas) + 1)]
        self.s_remap[0] = VOCAB.symbol_to_idx(VOCAB.UNK)
        for i, (a, _) in enumerate(aas):
            original_idx = VOCAB.symbol_to_idx(a) 
            self.s_map[original_idx] = i + 1 # start from 1
            self.s_remap[i + 1] =  original_idx
        self.s_map = nn.Parameter(torch.tensor(self.s_map, dtype=torch.long), requires_grad=False)
        self.s_remap = nn.Parameter(torch.tensor(self.s_remap, dtype=torch.long), requires_grad=False)
        
        if self.mode != 'fixseq':
            self.seq_linear = nn.Linear(hidden_size, len(self.s_remap))
        
    
    @torch.no_grad()
    def prepare_inputs(self, X, S, mask, atom_mask, lengths):

        # batch ids
        batch_ids = self.get_batch_ids(S, lengths)

        # edges
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)

        is_ctx = mask[row] == mask[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]

        return ctx_edges, inter_edges, batch_ids
    
    @torch.no_grad()
    def get_batch_ids(self, S, lengths):
        batch_ids = torch.zeros_like(S)
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        return batch_ids

    def rsample(self, H, Z, Z_centers, no_randomness=False):
        '''
            H: [N, latent_size]
            Z: [N, latent_channel, 3]
            Z_centers: [N, latent_channel, 3]
        '''

        if self.mode != 'fixseq':
            data_size = H.shape[0]
            H_mean = self.W_mean(H)
            H_log_var = -torch.abs(self.W_log_var(H)) #Following Mueller et al., z_log_var is log(\sigma^2)
            H_kl_loss = -0.5 * torch.sum(1.0 + H_log_var - H_mean * H_mean - torch.exp(H_log_var)) / data_size
            H_vecs = H_mean if no_randomness else H_mean + torch.exp(H_log_var / 2) * torch.randn_like(H_mean)
        else:
            H_vecs, H_kl_loss = None, 0

        if self.mode != 'fixbb':
            data_size = Z.shape[0]
            Z_mean_delta = Z - Z_centers
            Z_log_var = -torch.abs(self.W_Z_log_var(H)).view(-1, self.latent_n_channel, 3)
            Z_kl_loss = -0.5 * torch.sum(1.0 + Z_log_var - math.log(self.coord_prior_var) - Z_mean_delta * Z_mean_delta / self.coord_prior_var - torch.exp(Z_log_var) / self.coord_prior_var) / data_size
            Z_vecs = Z if no_randomness else Z + torch.exp(Z_log_var / 2) * torch.randn_like(Z)
        else:
            Z_vecs, Z_kl_loss = None, 0

        return H_vecs, Z_vecs, H_kl_loss, Z_kl_loss

    def _get_latent_channels(self, X, atom_mask):
        atom_weights = atom_mask.float() # 1 for atom, 0 for padding/missing, [N, 14]
        if hasattr(self, 'fix_alpha_carbon') and self.fix_alpha_carbon:
            return X[:, self.ca_channel_idx].unsqueeze(1) # use alpha carbon as latent channel
        elif self.latent_n_channel == 1:
            X = (X * atom_weights.unsqueeze(-1)).sum(1) # [N, 3]
            X = X / atom_weights.sum(-1).unsqueeze(-1) # [N, 3]
            return X.unsqueeze(1)
        elif self.latent_n_channel == 5:
            bb_X = X[:, :4]
            X = (X * atom_weights.unsqueeze(-1)).sum(1) # [N, 3]
            X = X / atom_weights.sum(-1).unsqueeze(-1) # [N, 3]
            X = torch.cat([bb_X, X.unsqueeze(1)], dim=1) # [N, 5, 3]
            return X
        else:
            raise NotImplementedError(f'Latent number of channels: {self.latent_n_channel} not implemented')

    def _get_latent_channel_anchors(self, X, atom_mask):
        if self.anchor_at_ca:
            return X[:, self.ca_channel_idx].unsqueeze(1)
        else:
            return self._get_latent_channels(X, atom_mask)
        
    def _fill_latent_channels(self, latent_X):
        if self.latent_n_channel == 1:
            return latent_X.repeat(1, self.n_channel, 1)
        elif self.latent_n_channel == 5:
            bb_X = latent_X[:, :4]
            sc_X = latent_X[:, 4].unsqueeze(1).repeat(1, self.n_channel - 4, 1)
            return torch.cat([bb_X, sc_X], dim=1)
        else:
            raise NotImplementedError(f'Latent number of channels: {self.latent_n_channel} not implemented')
        
    def _remove_sidechain_atom_mask(self, atom_mask, mask_generate):
        atom_mask = atom_mask.clone()
        bb_mask = atom_mask[mask_generate]
        bb_mask[:, 4:] = 0 # only backbone atoms are visible
        atom_mask[mask_generate] = bb_mask
        return atom_mask

    @torch.no_grad()
    def _mask_pep(self, S, atom_mask, mask_generate):
        assert self.mask_ratio > 0
        S, atom_mask = S.clone(), atom_mask.clone()
        pep_S = S[mask_generate]
        do_mask = torch.rand_like(pep_S, dtype=torch.float) < self.mask_ratio
        pep_S[do_mask] = self.mask_id

        S[mask_generate] = pep_S
        atom_mask[mask_generate ]= self._remove_sidechain_atom_mask(atom_mask[mask_generate], do_mask)

        return S, atom_mask

    def encode(self, X, S, mask, position_ids, lengths, atom_mask, no_randomness=False):
        true_X = X.clone()

        ctx_edges, inter_edges, batch_ids = self.prepare_inputs(X, S, mask, atom_mask, lengths)
        H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids)

        edges = torch.cat([ctx_edges, inter_edges], dim=1)
        atom_weights = atom_mask.float() # 1 for atom, 0 for padding/missing, [N, 14]

        H, pred_X = self.encoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_weights)
        H = H[mask]

        if self.mode != 'fixbb':
            if hasattr(self, 'fix_alpha_carbon') and self.fix_alpha_carbon:
                Z = self._get_latent_channels(true_X, atom_mask)
            else:
                Z = self._get_latent_channels(pred_X, atom_mask)
            Z_centers = self._get_latent_channel_anchors(true_X, atom_mask)
            Z, Z_centers = Z[mask], Z_centers[mask]
        else:
            Z, Z_centers = None, None

        # resample
        latent_H, latent_X, H_kl_loss, X_kl_loss = self.rsample(H, Z, Z_centers, no_randomness)
        return latent_H, latent_X, H_kl_loss, X_kl_loss
    
    def decode(self, X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing):
        X, S, atom_mask = X.clone(), S.clone(), atom_mask.clone()
        true_S = S[mask].clone()
        if self.mode != 'fixbb':  # fill coordinates with latent points
            X[mask] = self._fill_latent_channels(Z)
        if self.mode != 'fixseq':  # fill sequences with mask token
            S[mask] = self.latent_id
            H_from_latent = self.latent2hidden(H)

        if self.mode == 'fixbb':  # only backbone atoms are visible
            atom_mask = self._remove_sidechain_atom_mask(atom_mask, mask)
        elif self.mode == 'codesign':  # all channels are visible when deciding the sequence (all dummy atoms)
            atom_mask[mask] = 1
        else:  # fixseq mode does not need to change atom mask
            pass

        ctx_edges, inter_edges, batch_ids = self.prepare_inputs(X, S, mask, atom_mask, lengths)
        edges = torch.cat([ctx_edges, inter_edges], dim=1)

        # decode sequence
        if self.mode != 'fixseq':
            H_0, (atom_embeddings, _) = self.aa_feature(S, position_ids)
            H_0 = H_0.clone()
            H_0[mask] = H_from_latent  # TODO: how about the position encoding
            H, _ = self.seq_decoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_mask.float())
            pred_S_logits = self.seq_linear(H[mask])  # [Ntgt, 21]
            S = S.clone()
            if teacher_forcing:  # teacher forcing
                S[mask] = true_S
            else: # inference
                S[mask] = self.s_remap[torch.argmax(pred_S_logits, dim=-1)]
        else:
            pred_S_logits = None

        # decode sidechain
        if self.mode != 'fixbb':
            H_0, (atom_embeddings, atom_weights) = self.aa_feature(S, position_ids)
            H_0 = H_0.clone()
            if self.mode != 'fixseq':
                H_0[mask] = self.merge_S_H(torch.cat([H_from_latent, H_0[mask]], dim=-1))
                # H_0[mask] = H_from_latent
            atom_mask = atom_mask.clone()
            atom_mask[mask] = atom_weights.bool()[mask] & atom_mask[mask] # reset atomic visibility of the reconstruction part with the decoded sequence
            _, pred_X = self.sidechain_decoder(H_0, X, edges, channel_attr=atom_embeddings, channel_weights=atom_mask.float())
            pred_X = pred_X[mask]
        else:
            pred_X = None

        return pred_S_logits, pred_X

    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, teacher_forcing=True):
        true_X, true_S = X[mask].clone(), S[mask].clone()
        
        # encode: H (N*d), Z (N*3)
        if self.mask_ratio > 0:
            input_S, input_atom_mask = self._mask_pep(S, atom_mask, mask)
        else:
            input_S, input_atom_mask = S, atom_mask
        H, Z, H_kl_loss, Z_kl_loss = self.encode(X, input_S, mask, position_ids, lengths, input_atom_mask)
        
        if self.mode != 'fixbb':
            coord_reg_loss = F.mse_loss(Z, self._get_latent_channel_anchors(true_X, atom_mask[mask]))
        else:
            coord_reg_loss = 0

        # decode: S (N), Z (N * 14 * 3) with atom mask
        recon_S_logits, recon_X = self.decode(X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing)

        # sequence reconstruction loss
        if self.mode != 'fixseq':
            seq_recon_loss = F.cross_entropy(recon_S_logits, self.s_map[true_S])
            # aar
            with torch.no_grad():
                aar = (torch.argmax(recon_S_logits, dim=-1) == self.s_map[true_S]).sum() / len(recon_S_logits)
        else:
            seq_recon_loss, aar = 0, 1.0

        # coordinates reconstruction loss
        if self.mode != 'fixbb':
            xloss_mask = atom_mask[mask]
            batch_ids = self.get_batch_ids(S, lengths)[mask]
            segment_ids = torch.ones_like(true_S, device=true_S.device, dtype=torch.long)
            if self.n_channel == 4:  # backbone only
                loss_profile = {}
            else:
                true_struct_profile = self.protein_feature.get_struct_profile(true_X, true_S, batch_ids, self.aa_feature, segment_ids, xloss_mask)
                recon_struct_profile = self.protein_feature.get_struct_profile(recon_X, true_S, batch_ids, self.aa_feature, segment_ids, xloss_mask)
                loss_profile = { key + '_loss': F.l1_loss(recon_struct_profile[key], true_struct_profile[key]) for key in recon_struct_profile }

            # mse
            xloss = F.mse_loss(recon_X[xloss_mask], true_X[xloss_mask])
            loss_profile['Xloss'] = xloss

            # CA mse
            ca_xloss_mask = xloss_mask[:, self.ca_channel_idx]
            ca_xloss = F.mse_loss(recon_X[:, self.ca_channel_idx][ca_xloss_mask], true_X[:, self.ca_channel_idx][ca_xloss_mask])
            loss_profile['ca_Xloss'] = ca_xloss

            struct_recon_loss = 0
            for name in loss_profile:
                struct_recon_loss = struct_recon_loss + self.coord_loss_weights[name] * loss_profile[name]
        else:
            struct_recon_loss, loss_profile = 0, {}

        recon_loss = (1 - self.coord_loss_ratio) * (seq_recon_loss + self.h_kl_weight * H_kl_loss) + \
                     self.coord_loss_ratio * (struct_recon_loss + self.z_kl_weight * Z_kl_loss)

        return recon_loss, (seq_recon_loss, aar), (struct_recon_loss, loss_profile), (H_kl_loss, Z_kl_loss, coord_reg_loss)
    
    def _reconstruct(self, X, S, mask, position_ids, lengths, atom_mask, given_laten_H=None, given_latent_X=None, allow_unk=False, optimize_sidechain=True, idealize=False, no_randomness=False):
        if given_laten_H is None and given_latent_X is None:
            # encode: H (N*d), Z (N*3)
            H, Z, _, _ = self.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=no_randomness)

        else:
            H, Z = given_laten_H, given_latent_X

        # decode: S (N), Z (N * 14 * 3) with atom mask
        recon_S_logits, recon_X = self.decode(X, S, H, Z, mask, position_ids, lengths, atom_mask, teacher_forcing=False)
        batch_ids = self.get_batch_ids(S, lengths)[mask]
        
        if self.mode != 'fixseq':
            if not allow_unk:
                recon_S_logits[:, 0] = float('-inf')

            # map aa index back
            recon_S = self.s_remap[torch.argmax(recon_S_logits, dim=-1)]
            # ppls
            snll_all = F.cross_entropy(recon_S_logits, torch.argmax(recon_S_logits, dim=-1), reduction='none')
            batch_ppls = scatter_mean(snll_all, batch_ids, dim=0)
        else:
            recon_S = S[mask]
            batch_ppls = torch.zeros(batch_ids.max() + 1, device=recon_X.device).float()

        if self.mode == 'fixseq' or (self.mode != 'fixbb' and idealize):
            # rectify backbone
            recon_X = self.backbone_model(recon_X, batch_ids)
            # rectify sidechain
            recon_X = self.sidechain_model(recon_X, recon_S, batch_ids, optimize_sidechain)
        
        return recon_X, recon_S, batch_ppls, batch_ids

    @torch.no_grad()
    def test(self, X, S, mask, position_ids, lengths, atom_mask, given_laten_H=None, given_latent_X=None, return_tensor=False, allow_unk=False, optimize_sidechain=True, idealize=False, n_iter=1):

        no_randomness = given_laten_H is not None # in reconstruction mode, with latent variable derived from diffusion model
        for i in range(n_iter):
            recon_X, recon_S, batch_ppls, batch_ids = self._reconstruct(X, S, mask, position_ids, lengths, atom_mask, given_laten_H, given_latent_X, allow_unk, optimize_sidechain, idealize, no_randomness)
            X, S = X.clone(), S.clone()
            if self.mode != 'fixbb':
                X[mask] = recon_X
            if self.mode != 'fixseq':
                S[mask] = recon_S
            given_laten_H, given_latent_X = None, None # let the model encode and decode for later iterations
        
        if return_tensor:
            return recon_X, recon_S, batch_ppls

        batch_X, batch_S = [], []
        batch_ppls = batch_ppls.tolist()
        for i, l in enumerate(lengths):
            cur_mask = batch_ids == i
            if self.mode == 'fixbb':
                batch_X.append(None)
            else:
                batch_X.append(recon_X[cur_mask].tolist())
            if self.mode == 'fixseq':
                batch_S.append(None)
            else:
                batch_S.append(''.join([VOCAB.idx_to_symbol(s) for s in recon_S[cur_mask]]))

        return batch_X, batch_S, batch_ppls
