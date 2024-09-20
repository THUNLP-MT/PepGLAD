#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.format import VOCAB

from utils.nn_utils import sequential_and
from utils import const


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings

# embedding of amino acids. (default: concat residue embedding and atom embedding to one vector)
class AminoAcidEmbedding(nn.Module):
    '''
    [residue embedding + position embedding, mean(atom embeddings + atom position embeddings)]
    '''
    def __init__(self, num_res_type, num_atom_type, num_atom_pos, res_embed_size, atom_embed_size,
                 atom_pad_id=VOCAB.get_atom_pad_idx(), max_position=256, relative_position=True):
        super().__init__()
        self.residue_embedding = nn.Embedding(num_res_type, res_embed_size)
        if relative_position:
            self.res_pos_embedding = SinusoidalPositionEmbedding(res_embed_size)  # relative positional encoding
        else:
            self.res_pos_embedding = nn.Embedding(max_position, res_embed_size)  # absolute position encoding
        self.atom_embedding = nn.Embedding(num_atom_type, atom_embed_size)
        self.atom_pos_embedding = nn.Embedding(num_atom_pos, atom_embed_size)
        self.atom_pad_id = atom_pad_id
        self.eps = 1e-10  # for mean of atom embedding (some residues have no atom at all)
    
    def forward(self, S, RP, A, AP):
        '''
        :param S: [N], residue types
        :param RP: [N], residue positions
        :param A: [N, n_channel], atom types
        :param AP: [N, n_channel], atom positions
        '''
        res_embed = self.residue_embedding(S) + self.res_pos_embedding(RP)  # [N, res_embed_size]
        atom_embed = self.atom_embedding(A) + self.atom_pos_embedding(AP)   # [N, n_channel, atom_embed_size]
        atom_not_pad = (AP != self.atom_pad_id)  # [N, n_channel]
        denom = torch.sum(atom_not_pad, dim=-1, keepdim=True) + self.eps
        atom_embed = torch.sum(atom_embed * atom_not_pad.unsqueeze(-1), dim=1) / denom  # [N, atom_embed_size]
        return torch.cat([res_embed, atom_embed], dim=-1)  # [N, res_embed_size + atom_embed_size]


class AminoAcidFeature(nn.Module):
    def __init__(self, backbone_only=False) -> None:
        super().__init__()

        self.backbone_only = backbone_only

        # number of classes
        self.num_aa_type = len(VOCAB)
        self.num_atom_type = VOCAB.get_num_atom_type()
        self.num_atom_pos = VOCAB.get_num_atom_pos()

        # atom-level special tokens
        self.atom_mask_idx = VOCAB.get_atom_mask_idx()
        self.atom_pad_idx = VOCAB.get_atom_pad_idx()
        self.atom_pos_mask_idx = VOCAB.get_atom_pos_mask_idx()
        self.atom_pos_pad_idx = VOCAB.get_atom_pos_pad_idx()
        
        self.mask_idx = VOCAB.get_mask_idx()
        self.unk_idx = VOCAB.symbol_to_idx(VOCAB.UNK)
        self.latent_idx = VOCAB.symbol_to_idx(VOCAB.LAT)

        # atoms encoding
        residue_atom_type, residue_atom_pos = [], []
        backbone = [VOCAB.atom_to_idx(atom[0]) for atom in VOCAB.backbone_atoms]
        backbone_pos = [VOCAB.atom_pos_to_idx(atom[1:]) for atom in VOCAB.backbone_atoms]
        n_channel = VOCAB.MAX_ATOM_NUMBER if not backbone_only else 4
        special_mask = VOCAB.get_special_mask()
        for i in range(len(VOCAB)):
            if i == self.mask_idx or i == self.unk_idx:
                # mask or unk
                residue_atom_type.append(backbone + [self.atom_mask_idx for _ in range(n_channel - len(backbone))])
                residue_atom_pos.append(backbone_pos + [self.atom_pos_mask_idx for _ in range(n_channel - len(backbone_pos))])
            elif i == self.latent_idx:
                # latent index
                residue_atom_type.append([VOCAB.get_atom_latent_idx() for _ in range(n_channel)])
                residue_atom_pos.append([VOCAB.get_atom_pos_latent_idx() for _ in range(n_channel)])
            elif special_mask[i] == 1:
                # other special token (pad)
                residue_atom_type.append([self.atom_pad_idx for _ in range(n_channel)])
                residue_atom_pos.append([self.atom_pos_pad_idx for _ in range(n_channel)])
            else:
                # normal amino acids
                atom_type, atom_pos = backbone, backbone_pos
                if not backbone_only:
                    sidechain_atoms = const.sidechain_atoms[VOCAB.idx_to_symbol(i)]
                    atom_type = atom_type + [VOCAB.atom_to_idx(atom[0]) for atom in sidechain_atoms]
                    atom_pos = atom_pos + [VOCAB.atom_pos_to_idx(atom[1]) for atom in sidechain_atoms]
                num_pad = n_channel - len(atom_type)
                residue_atom_type.append(atom_type + [self.atom_pad_idx for _ in range(num_pad)])
                residue_atom_pos.append(atom_pos + [self.atom_pos_pad_idx for _ in range(num_pad)])
        
        # mapping from residue to atom types and positions
        self.residue_atom_type = nn.parameter.Parameter(
            torch.tensor(residue_atom_type, dtype=torch.long),
            requires_grad=False)
        self.residue_atom_pos = nn.parameter.Parameter(
            torch.tensor(residue_atom_pos, dtype=torch.long),
            requires_grad=False)

        # sidechain geometry
        if not backbone_only:
            sc_bonds, sc_bonds_mask = [], []
            sc_chi_atoms, sc_chi_atoms_mask = [], []
            for i in range(len(VOCAB)):
                if special_mask[i] == 1:
                    sc_bonds.append([])
                    sc_chi_atoms.append([])
                else:
                    symbol = VOCAB.idx_to_symbol(i)
                    atom_type = VOCAB.backbone_atoms + const.sidechain_atoms[symbol]
                    atom2channel = { atom: i for i, atom in enumerate(atom_type) }
                    chi_atoms = const.chi_angles_atoms[VOCAB.symbol_to_abrv(symbol)]
                    bond_atoms = const.sidechain_bonds[symbol]
                    sc_chi_atoms.append(
                        [[atom2channel[atom] for atom in atoms] for atoms in chi_atoms]
                    )
                    bonds = []
                    for src_atom, dst_atom, _ in bond_atoms:
                        bonds.append((atom2channel[src_atom], atom2channel[dst_atom]))
                    sc_bonds.append(bonds)
            max_num_chis = max([len(chis) for chis in sc_chi_atoms])
            max_num_bonds = max([len(bonds) for bonds in sc_bonds])
            for i in range(len(VOCAB)):
                num_chis, num_bonds = len(sc_chi_atoms[i]), len(sc_bonds[i])
                num_pad_chis, num_pad_bonds = max_num_chis - num_chis, max_num_bonds - num_bonds
                sc_chi_atoms_mask.append(
                    [1 for _ in range(num_chis)] + [0 for _ in range(num_pad_chis)]
                )
                sc_bonds_mask.append(
                    [1 for _ in range(num_bonds)] + [0 for _ in range(num_pad_bonds)]
                )
                sc_chi_atoms[i].extend([[-1, -1, -1, -1] for _ in range(num_pad_chis)])
                sc_bonds[i].extend([(-1, -1) for _ in range(num_pad_bonds)])

            # mapping residues to their sidechain chi angle atoms and bonds
            self.sidechain_chi_angle_atoms = nn.parameter.Parameter(
                torch.tensor(sc_chi_atoms, dtype=torch.long),
                requires_grad=False)
            self.sidechain_chi_mask = nn.parameter.Parameter(
                torch.tensor(sc_chi_atoms_mask, dtype=torch.bool),
                requires_grad=False
            )
            self.sidechain_bonds = nn.parameter.Parameter(
                torch.tensor(sc_bonds, dtype=torch.long),
                requires_grad=False
            )
            self.sidechain_bonds_mask = nn.parameter.Parameter(
                torch.tensor(sc_bonds_mask, dtype=torch.bool),
                requires_grad=False
            )

    def _construct_atom_type(self, S):
        # construct atom types
        return self.residue_atom_type[S]
    
    def _construct_atom_pos(self, S):
        # construct atom positions
        return self.residue_atom_pos[S]

    @torch.no_grad()
    def get_sidechain_chi_angles_atoms(self, S):
        chi_angles_atoms = self.sidechain_chi_angle_atoms[S]  # [N, max_num_chis, 4]
        chi_mask = self.sidechain_chi_mask[S]  # [N, max_num_chis]
        return chi_angles_atoms, chi_mask

    @torch.no_grad()
    def get_sidechain_bonds(self, S):
        bonds = self.sidechain_bonds[S]  # [N, max_num_bond, 2]
        bond_mask = self.sidechain_bonds_mask[S]
        return bonds, bond_mask


class SeparatedAminoAcidFeature(AminoAcidFeature):
    '''
    Separate embeddings of atoms and residues
    '''
    def __init__(
            self,
            embed_size,
            atom_embed_size,
            max_position,
            relative_position=True,
            fix_atom_weights=False,
            backbone_only=False
        ) -> None:
        super().__init__(backbone_only=backbone_only)
        atom_weights_mask = self.residue_atom_type == self.atom_pad_idx
        self.register_buffer('atom_weights_mask', atom_weights_mask)
        self.fix_atom_weights = fix_atom_weights
        if fix_atom_weights:
            atom_weights = torch.ones_like(self.residue_atom_type, dtype=torch.float)
        else:
            atom_weights = torch.randn_like(self.residue_atom_type, dtype=torch.float)
        atom_weights[atom_weights_mask] = 0
        self.atom_weight = nn.parameter.Parameter(atom_weights, requires_grad=not fix_atom_weights)
        self.zero_atom_weight = nn.parameter.Parameter(torch.zeros_like(atom_weights), requires_grad=False)
        
        self.aa_embedding = AminoAcidEmbedding(
            self.num_aa_type, self.num_atom_type, self.num_atom_pos,
            embed_size, atom_embed_size, self.atom_pad_idx,
            max_position, relative_position)
    
    def get_atom_weights(self, residue_types):
        weights = torch.where(
            self.atom_weights_mask,
            self.zero_atom_weight,
            self.atom_weight
        )  # [num_aa_classes, max_atom_number(n_channel)]
        if not self.fix_atom_weights:
            weights = F.normalize(weights, dim=-1)
        return weights[residue_types]

    def forward(self, S, position_ids, smooth_prob=None, smooth_mask=None):
        atom_type = self.residue_atom_type[S]  # [N, n_channel]
        atom_pos = self.residue_atom_pos[S]     # [N, n_channel]

        # residue embedding
        pos_embedding = self.aa_embedding.res_pos_embedding(position_ids)
        H = self.aa_embedding.residue_embedding(S)
        if smooth_prob is not None:
            res_embeddings = self.aa_embedding.residue_embedding(
                torch.arange(smooth_prob.shape[-1], device=S.device, dtype=S.dtype)
            )  # [num_aa_type, embed_size]
            H[smooth_mask] = smooth_prob.mm(res_embeddings)
        H = H + pos_embedding

        # atom embedding
        atom_embedding = self.aa_embedding.atom_embedding(atom_type) +\
                         self.aa_embedding.atom_pos_embedding(atom_pos)
        atom_weights = self.get_atom_weights(S)
        
        return H, (atom_embedding, atom_weights)


class ProteinFeature:
    def __init__(self, backbone_only=False):
        self.backbone_only = backbone_only

    def _cal_sidechain_bond_lengths(self, S, X, aa_feature: AminoAcidFeature, atom_mask=None):
        bonds, bonds_mask = aa_feature.get_sidechain_bonds(S)
        n = torch.nonzero(bonds_mask)[:, 0]  # [Nbonds]
        src, dst = bonds[bonds_mask].T
        src_X, dst_X = X[(n, src)], X[(n, dst)]  # [Nbonds, 3]
        bond_lengths = torch.norm(dst_X - src_X, dim=-1)
        if atom_mask is not None:
            mask = atom_mask[(n, src)] & atom_mask[(n, dst)]
            bond_lengths = bond_lengths[mask]
        return bond_lengths

    def _cal_sidechain_chis(self, S, X, aa_feature: AminoAcidFeature, atom_mask=None):
        chi_atoms, chi_mask = aa_feature.get_sidechain_chi_angles_atoms(S)
        n = torch.nonzero(chi_mask)[:, 0]  # [Nchis]
        a0, a1, a2, a3 = chi_atoms[chi_mask].T  # [Nchis]
        x0, x1, x2, x3 = X[(n, a0)], X[(n, a1)], X[(n, a2)], X[(n, a3)]  # [Nchis, 3]
        u_0, u_1, u_2 = (x1 - x0), (x2 - x1), (x3 - x2)  # [Nchis, 3]
        # normals of the two planes
        n_1 = F.normalize(torch.cross(u_0, u_1), dim=-1)  # [Nchis, 3]
        n_2 = F.normalize(torch.cross(u_1, u_2), dim=-1)  # [Nchis, 3]
        cosChi = (n_1 * n_2).sum(-1)  # [Nchis]
        eps = 1e-7
        cosChi = torch.clamp(cosChi, -1 + eps, 1 - eps)
        if atom_mask is not None:
            mask = atom_mask[(n, a0)] & atom_mask[(n, a1)] & atom_mask[(n, a2)] & atom_mask[(n, a3)]
            cosChi = cosChi[mask]
        return cosChi

    def _cal_backbone_bond_lengths(self, X, batch_ids, segment_ids, atom_mask=None):
        # loss of backbone (...N-CA-C(O)-N...) bond length
        # N-CA, CA-C, C=O
        bl1 = torch.norm(X[:, 1:4] - X[:, :3], dim=-1)  # [N, 3], (N-CA), (CA-C), (C=O)
        if atom_mask is not None:
            bl1 = bl1[atom_mask[:, 1:4] & atom_mask[:, :3]]
        else:
            bl1 = bl1.flatten()
        # C-N
        bl2 = torch.norm(X[1:, 0] - X[:-1, 2], dim=-1)  # [N-1]
        same_chain_mask = (segment_ids[1:] == segment_ids[:-1]) & (batch_ids[1:] == batch_ids[:-1])
        if atom_mask is not None:
            mask = atom_mask[1:, 0] & atom_mask[:-1, 2] & same_chain_mask
        else:
            mask = same_chain_mask
        bl2 = bl2[mask]
        bl = torch.cat([bl1, bl2], dim=0)
        return bl

    def _cal_backbone_dihedral_angles(self, X, batch_ids, segment_ids, atom_mask=None):
        ori_X = X.clone() # used for calculating bond angles
        X = X[:, :3].reshape(-1, 3)  # [N * 3, 3], N, CA, C
        U = F.normalize(X[1:] - X[:-1], dim=-1)  # [N * 3 - 1, 3]

        # 1. dihedral angles
        u_2, u_1, u_0 = U[:-2], U[1:-1], U[2:]   # [N * 3 - 3, 3]
        # backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        # angle between normals
        eps = 1e-7
        cosD = (n_2 * n_1).sum(-1)  # [(N-1) * 3]
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        # D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        seg_id_atom = segment_ids.repeat(1, 3).flatten()  # [N * 3]
        batch_id_atom = batch_ids.repeat(1, 3).flatten()
        same_chain_mask = sequential_and(
            seg_id_atom[:-3] == seg_id_atom[1:-2],
            seg_id_atom[1:-2] == seg_id_atom[2:-1],
            seg_id_atom[2:-1] == seg_id_atom[3:],
            batch_id_atom[:-3] == batch_id_atom[1:-2],
            batch_id_atom[1:-2] == batch_id_atom[2:-1],
            batch_id_atom[2:-1] == batch_id_atom[3:]
        )  # [N * 3 - 3]
        # D = D[same_chain_mask]
        if atom_mask is not None:
            mask = atom_mask[:, :3].flatten()  # [N * 3]
            mask = mask[1:] & mask[:-1] # [N * 3 - 1]
            mask = mask[:-2] & mask[1:-1] & mask[2:] # [N * 3 - 3]
            mask = mask & same_chain_mask
        else:
            mask = same_chain_mask

        cosD = cosD[mask]

        # # 2. bond angles (C_{n-1}-N, N-CA), (N-CA, CA-C), (CA-C, C=O), (CA-C, C-N_{n+1}), (O=C, C-Nn)
        # u_0, u_1 = U[:-1], U[1:]  # [N*3 - 2, 3]
        # cosA1 = ((-u_0) * u_1).sum(-1)  # [N*3 - 2], (C_{n-1}-N, N-CA), (N-CA, CA-C), (CA-C, C-N_{n+1})
        # same_chain_mask = sequential_and(
        #     seg_id_atom[:-2] == seg_id_atom[1:-1],
        #     seg_id_atom[1:-1] == seg_id_atom[2:]
        # )
        # cosA1 = cosA1[same_chain_mask]  # [N*3 - 2 * num_chain]
        # u_co = F.normalize(ori_X[:, 3] - ori_X[:, 2], dim=-1)  # [N, 3], C=O
        # u_cca = -U[1::3]  # [N, 3], C-CA
        # u_cn = U[2::3] # [N-1, 3], C-N_{n+1}
        # cosA2 = (u_co * u_cca).sum(-1)  # [N], (C=O, C-CA)
        # cosA3 = (u_co[:-1] * u_cn).sum(-1)  # [N-1], (C=O, C-N_{n+1})
        # same_chain_mask = (seg_id[:-1] == seg_id[1:]) # [N-1]
        # cosA3 = cosA3[same_chain_mask]
        # cosA = torch.cat([cosA1, cosA2, cosA3], dim=-1)
        # cosA = torch.clamp(cosA, -1 + eps, 1 - eps)

        # return cosD, cosA
        return cosD
    
    def get_struct_profile(self, X, S, batch_ids, aa_feature: AminoAcidFeature, segment_ids=None, atom_mask=None):
        '''
            X: [N, 14, 3], coordinates of all atoms
            batch_ids: [N], indicate which item the residue belongs to
            segment_ids: [N], indicate which chain the residue belongs to
            aa_feature: AminoAcidFeature, storing geometric constants
            atom_mask: [N, 14], 0 for padding/missing
        '''
        if segment_ids is None:  # default regarded as monomers
            segment_ids = torch.ones_like(batch_ids)
        return {
            'bb_bond_lengths': self._cal_backbone_bond_lengths(X, batch_ids, segment_ids, atom_mask),
            'sc_bond_lengths': self._cal_sidechain_bond_lengths(S, X, aa_feature, atom_mask),
            'bb_dihedral_angles': self._cal_backbone_dihedral_angles(X, batch_ids, segment_ids, atom_mask),
            'sc_chi_angles': self._cal_sidechain_chis(S, X, aa_feature, atom_mask)
        }