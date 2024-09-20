#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

import numpy as np

from data.format import VOCAB, Block
from utils import const


def blocks_to_data(*blocks_list: List[List[Block]]):
    B, A, X, atom_positions, block_lengths, segment_ids = [], [], [], [], [], []
    atom_mask, is_ca = [], []
    topo_edge_index, topo_edge_attr, atom_names = [], [], []
    last_c_node_id = None
    for i, blocks in enumerate(blocks_list):
        if len(blocks) == 0:
            continue
        cur_B, cur_A, cur_X, cur_atom_positions, cur_block_lengths = [], [], [], [], []
        cur_atom_mask, cur_is_ca = [], []
        # other nodes
        for block in blocks:
            b, symbol = VOCAB.abrv_to_idx(block.abrv), VOCAB.abrv_to_symbol(block.abrv)
            x, a, positions, m, ca = [], [], [], [], []
            atom2node_id = {}
            if symbol == '?':
                atom_missing = {}
            else:
                atom_missing = { atom_name: True for atom_name in const.backbone_atoms + const.sidechain_atoms[symbol] }
            for atom in block:
                atom2node_id[atom.name] = len(A) + len(cur_A) + len(a)
                a.append(VOCAB.atom_to_idx(atom.get_element()))
                x.append(atom.get_coord())
                pos_code = ''.join((c for c in atom.get_pos_code() if not c.isdigit()))
                positions.append(VOCAB.atom_pos_to_idx(pos_code))
                if atom.name in atom_missing:
                    atom_missing[atom.name] = False
                m.append(1)
                ca.append(atom.name == 'CA')
                atom_names.append(atom.name)
            for atom_name in atom_missing:
                if atom_missing[atom_name]:
                    atom2node_id[atom_name] = len(A) + len(cur_A) + len(a)
                    a.append(VOCAB.atom_to_idx(atom_name[0])) # only C, N, O, S in proteins
                    x.append([0, 0, 0])
                    pos_code = ''.join((c for c in atom_name[1:] if not c.isdigit()))
                    positions.append(VOCAB.atom_pos_to_idx(pos_code))
                    m.append(0)
                    ca.append(atom_name == 'CA')
                    atom_names.append(atom_name)
            block_len = len(a)
            cur_B.append(b)
            cur_A.extend(a)
            cur_X.extend(x)
            cur_atom_positions.extend(positions)
            cur_block_lengths.append(block_len)
            cur_atom_mask.extend(m)
            cur_is_ca.extend(ca)

            # topology edges
            for src, dst, bond_type in const.sidechain_bonds.get(VOCAB.abrv_to_symbol(block.abrv), []):
                src, dst = atom2node_id[src], atom2node_id[dst]
                topo_edge_index.append((src, dst))  # no direction
                topo_edge_index.append((dst, src))
                topo_edge_attr.append(bond_type)
                topo_edge_attr.append(bond_type)
            if last_c_node_id is not None and ('CA' in atom2node_id):
                src, dst = last_c_node_id, atom2node_id['N']
                topo_edge_index.append((src, dst))  # no direction
                topo_edge_index.append((dst, src))
                topo_edge_attr.append(4)
                topo_edge_attr.append(4)
            if 'CA' not in atom2node_id:
                last_c_node_id = None
            else:
                last_c_node_id = atom2node_id['C']
            
        # update coordinates of the global node to the center
        # cur_X[0] = np.mean(cur_X[1:], axis=0)
        cur_segment_ids = [i for _ in cur_B]
        
        # finish these blocks
        B.extend(cur_B)
        A.extend(cur_A)
        X.extend(cur_X)
        atom_positions.extend(cur_atom_positions)
        block_lengths.extend(cur_block_lengths)
        segment_ids.extend(cur_segment_ids)
        atom_mask.extend(cur_atom_mask)
        is_ca.extend(cur_is_ca)

    X = np.array(X).tolist()
    topo_edge_index = np.array(topo_edge_index).T.tolist()
    topo_edge_attr = (np.array(topo_edge_attr) - 1).tolist() # type starts from 0 but bond type starts from 1
        
    data = {
        'X': X,             # [Natom, 2, 3]
        'B': B,             # [Nb], block (residue) type
        'A': A,             # [Natom]
        'atom_positions': atom_positions,  # [Natom]
        'block_lengths': block_lengths,  # [Nresidue]
        'segment_ids': segment_ids,      # [Nresidue]
        'atom_mask': atom_mask,          # [Natom]
        'is_ca': is_ca,                  # [Natom]
        'atom_names': atom_names,        # [Natom]
        'topo_edge_index': topo_edge_index, # atom level
        'topo_edge_attr': topo_edge_attr
    }

    return data