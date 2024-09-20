#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import argparse
from tqdm import tqdm
from os.path import splitext, basename

import ray
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.format import Atom, Block, VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from data.codesign import calculate_covariance_matrix
from utils.const import sidechain_atoms
from utils.logger import print_log
from evaluation.dG.openmm_relaxer import ForceFieldMinimizer


class DesignDataset(torch.utils.data.Dataset):

    MAX_N_ATOM = 14

    def __init__(self, pdbs, epitopes, lengths_range=None, seqs=None) -> None:
        super().__init__()
        self.pdbs = pdbs
        self.epitopes = epitopes
        self.lengths_range = lengths_range
        self.seqs = seqs
        # structure prediction or codesign
        assert (self.seqs is not None and self.lengths_range is None) | \
               (self.seqs is None and self.lengths_range is not None)

    def get_epitope(self, idx):
        pdb, epitope_def = self.pdbs[idx], self.epitopes[idx]

        with open(epitope_def, 'r') as fin:
            epitope = json.load(fin)
        to_str = lambda pos: f'{pos[0]}-{pos[1]}'
        epi_map = {}
        for chain_name, pos in epitope:
            if chain_name not in epi_map:
                epi_map[chain_name] = {}
            epi_map[chain_name][to_str(pos)] = True
        residues, position_ids = [], []
        chain2blocks = pdb_to_list_blocks(pdb, list(epi_map.keys()), dict_form=True)
        if len(chain2blocks) != len(epi_map):
            print_log(f'Some chains in the epitope are missing. Parsed {list(chain2blocks.keys())}, given {list(epi_map.keys())}.', level='WARN')
        for chain_name in chain2blocks:
            chain = chain2blocks[chain_name]
            for i, block in enumerate(chain):  # residue
                if to_str(block.id) in epi_map[chain_name]:
                    residues.append(block)
                    position_ids.append(i + 1) # position ids start from 1
        return residues, position_ids, chain2blocks

    def generate_pep_chain(self, idx):
        if self.lengths_range is not None: # codesign
            lmin, lmax = self.lengths_range[idx]
            length = np.random.randint(lmin, lmax)
            unk_block = Block(VOCAB.symbol_to_abrv(VOCAB.UNK), [Atom('CA', [0, 0, 0], 'C')])
            return [unk_block] * length
        else:
            seq = self.seqs[idx]
            blocks = []
            for s in seq:
                atoms = []
                for atom_name in VOCAB.backbone_atoms + sidechain_atoms.get(s, []):
                    atoms.append(Atom(atom_name, [0, 0, 0], atom_name[0]))
                blocks.append(Block(VOCAB.symbol_to_abrv(s), atoms))
            return blocks
    
    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx: int):
        rec_blocks, rec_position_ids, rec_chain2blocks = self.get_epitope(idx)
        lig_blocks = self.generate_pep_chain(idx)

        mask = [0 for _ in rec_blocks] + [1 for _ in lig_blocks]
        position_ids = rec_position_ids + [i + 1 for i, _ in enumerate(lig_blocks)]
        X, S, atom_mask = [], [], []
        for block in rec_blocks + lig_blocks:
            symbol = VOCAB.abrv_to_symbol(block.abrv)
            atom2coord = { unit.name: unit.get_coord() for unit in block.units }
            bb_pos = np.mean(list(atom2coord.values()), axis=0).tolist()
            coords, coord_mask = [], []
            for atom_name in VOCAB.backbone_atoms + sidechain_atoms.get(symbol, []):
                if atom_name in atom2coord:
                    coords.append(atom2coord[atom_name])
                    coord_mask.append(1)
                else:
                    coords.append(bb_pos)
                    coord_mask.append(0)
            n_pad = self.MAX_N_ATOM - len(coords)
            for _ in range(n_pad):
                coords.append(bb_pos)
                coord_mask.append(0)

            X.append(coords)
            S.append(VOCAB.symbol_to_idx(symbol))
            atom_mask.append(coord_mask)
        
        X, atom_mask = torch.tensor(X, dtype=torch.float), torch.tensor(atom_mask, dtype=torch.bool)
        mask = torch.tensor(mask, dtype=torch.bool)
        cov = calculate_covariance_matrix(X[~mask][:, 1][atom_mask[~mask][:, 1]].numpy()) # only use the receptor to derive the affine transformation
        eps = 1e-4
        cov = cov + eps * np.identity(cov.shape[0])
        L = torch.from_numpy(np.linalg.cholesky(cov)).float().unsqueeze(0)

        return {
            'X': X,                                                         # [N, 14] or [N, 4] if backbone_only == True
            'S': torch.tensor(S, dtype=torch.long),                         # [N]
            'position_ids': torch.tensor(position_ids, dtype=torch.long),   # [N]
            'mask': mask,                                                   # [N], 1 for generation
            'atom_mask': atom_mask,                                         # [N, 14] or [N, 4], 1 for having records in the PDB
            'lengths': len(S),
            'rec_chain2blocks': rec_chain2blocks,
            'L': L
        }

    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'rec_chain2blocks':
                results[key] = values
            else:
                results[key] = torch.cat(values, dim=0)
        return results


@ray.remote(num_cpus=1, num_gpus=1/16)
def openmm_relax(pdb_path):
    force_field = ForceFieldMinimizer()
    force_field(pdb_path, pdb_path)
    return pdb_path


def design(mode, ckpt, gpu, pdbs, epitope_defs, n_samples, out_dir,
           lengths_range=None, seqs=None, identifiers=None, batch_size=8, num_workers=4):

    # create out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    result_summary = open(os.path.join(out_dir, 'summary.jsonl'), 'w')
    if identifiers is None:
        identifiers = [splitext(basename(pdb))[0] for pdb in pdbs]
    # load model
    device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')
    model = torch.load(ckpt, map_location='cpu')
    model.to(device)
    model.eval()

    # generate dataset
    # expand data
    if lengths_range is None: lengths_range = [None for _ in pdbs]
    if seqs is None: seqs = [None for _ in pdbs]
    expand_pdbs, expand_epitopes, expand_lens, expand_ids, expand_seqs = [], [], [], [], []
    for _id, pdb, epitope, l, s, n in zip(identifiers, pdbs, epitope_defs, lengths_range, seqs, n_samples):
        expand_ids.extend([f'{_id}_{i}' for i in range(n)])
        expand_pdbs.extend([pdb for _ in range(n)])
        expand_epitopes.extend([epitope for _ in range(n)])
        expand_lens.extend([l for _ in range(n)])
        expand_seqs.extend([s for _ in range(n)])
    # create dataset
    if expand_lens[0] is None: expand_lens = None
    if expand_seqs[0] is None: expand_seqs = None
    dataset = DesignDataset(expand_pdbs, expand_epitopes, expand_lens, expand_seqs)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=dataset.collate_fn,
                             shuffle=False
                            )
    
    # generate peptides
    cnt = 0
    all_pdbs = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # generate
            batch_X, batch_S, batch_pmetric = model.sample(
                batch['X'], batch['S'],
                batch['mask'], batch['position_ids'],
                batch['lengths'], batch['atom_mask'],
                L=batch['L'], sample_opt={
                    'energy_func': 'default',
                    'energy_lambda': 0.5 if mode == 'struct_pred' else 0.8
                }
            )
        # save data
        for X, S, pmetric, rec_chain2blocks in zip(batch_X, batch_S, batch_pmetric, batch['rec_chain2blocks']):
            if S is None: S = expand_seqs[cnt] # structure prediction
            lig_blocks = []
            for x, s in zip(X, S):
                abrv = VOCAB.symbol_to_abrv(s)
                atoms = VOCAB.backbone_atoms + sidechain_atoms[VOCAB.abrv_to_symbol(abrv)]
                units = [
                    Atom(atom_name, coord, atom_name[0]) for atom_name, coord in zip(atoms, x)
                ]
                lig_blocks.append(Block(abrv, units))
            list_blocks, chain_names = [], []
            for chain in rec_chain2blocks:
                list_blocks.append(rec_chain2blocks[chain])
                chain_names.append(chain)
            pep_chain_id = chr(max([ord(c) for c in chain_names]) + 1)
            list_blocks.append(lig_blocks)
            chain_names.append(pep_chain_id)
            out_pdb = os.path.join(out_dir, expand_ids[cnt] + '.pdb')
            list_blocks_to_pdb(list_blocks, chain_names, out_pdb)
            all_pdbs.append(out_pdb)
            result_summary.write(json.dumps({
                'id': expand_ids[cnt],
                'rec_chains': list(rec_chain2blocks.keys()),
                'pep_chain': pep_chain_id,
                'pep_seq': ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in lig_blocks])
            }) + '\n')
            result_summary.flush()
            cnt += 1
    result_summary.close()

    print_log(f'Running openmm relaxation...')
    ray.init(num_cpus=8)
    futures = [openmm_relax.remote(path) for path in all_pdbs]
    pbar = tqdm(total=len(futures))
    while len(futures) > 0:
       done_ids, futures = ray.wait(futures, num_returns=1)
       for done_id in done_ids:
            done_path = ray.get(done_id)
            pbar.update(1)
    print_log(f'Done')


def parse():
    parser = argparse.ArgumentParser(description='run pepglad for codesign or structure prediction')
    parser.add_argument('--mode', type=str, required=True, choices=['codesign', 'struct_pred'], help='Running mode')
    parser.add_argument('--pdb', type=str, required=True, help='Path to the PDB file of the target protein')
    parser.add_argument('--pocket', type=str, required=True, help='Path to the pocket definition (*.json generated by detect_pocket)')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--peptide_seq', type=str, required='struct_pred' in sys.argv, help='Peptide sequence for structure prediction')
    parser.add_argument('--length_min', type=int, required='codesign' in sys.argv, help='Minimum peptide length for codesign (inclusive)')
    parser.add_argument('--length_max', type=int, required='codesign' in sys.argv, help='Maximum peptide length for codesign (exclusive)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    proj_dir = os.path.join(os.path.dirname(__file__), '..')
    ckpt = os.path.join(proj_dir, 'checkpoints', 'fixseq.ckpt' if args.mode == 'struct_pred' else 'codesign.ckpt')
    print_log(f'Loading checkpoint: {ckpt}')
    design(
        mode=args.mode,
        ckpt=ckpt,                          # path to the checkpoint of the trained model
        gpu=args.gpu,                       # the ID of the GPU to use
        pdbs=[args.pdb],                    # paths to the PDB file of each antigen
        epitope_defs=[args.pocket],         # paths to the epitope (pocket) definitions
        n_samples=[args.n_samples],         # number of samples for each epitope
        out_dir=args.out_dir,               # output directory
        identifiers=[os.path.basename(os.path.splitext(args.pdb)[0])], # file name (name of each output candidate)
        lengths_range=[(args.length_min, args.length_max)] if args.mode == 'codesign' else None,    # range of acceptable peptide lengths, left inclusive, right exclusive
        seqs=[args.peptide_seq] if args.mode == 'struct_pred' else None                             # peptide sequences for structure prediction
    )