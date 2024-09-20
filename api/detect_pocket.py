#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import numpy as np

from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_cb_interface, dist_matrix_from_blocks


def get_interface(pdb, receptor_chains, ligand_chains, pocket_th=10.0):  # CB distance
    list_blocks, chain_ids = pdb_to_list_blocks(pdb, receptor_chains + ligand_chains, return_chain_ids=True)
    chain2blocks = {chain: block for chain, block in zip(chain_ids, list_blocks)}
    for c in receptor_chains:
        assert c in chain2blocks, f'Chain {c} not found for receptor'
    for c in ligand_chains:
        assert c in chain2blocks, f'Chain {c} not found for ligand'

    rec_blocks, rec_block_chains, lig_blocks, lig_block_chains = [], [], [], []
    for c in receptor_chains:
        for block in chain2blocks[c]:
            rec_blocks.append(block)
            rec_block_chains.append(c)
    for c in ligand_chains:
        for block in chain2blocks[c]:
            lig_blocks.append(block)
            lig_block_chains.append(c)

    _, (pocket_idx, lig_if_idx) = blocks_cb_interface(rec_blocks, lig_blocks, pocket_th)  # 10A for pocket size based on CB
    epitope = []
    for i in pocket_idx:
        epitope.append((rec_blocks[i], rec_block_chains[i], i))

    dist_mat = dist_matrix_from_blocks([rec_blocks[i] for i in pocket_idx], [lig_blocks[i] for i in lig_if_idx])
    min_dists = np.min(dist_mat, axis=-1)  # [Nrec]
    lig_idxs = np.argmin(dist_mat, axis=-1)  # [Nrec]
    dists = []
    for i, d in zip(lig_idxs, min_dists):
        i = lig_if_idx[i]
        dists.append((lig_blocks[i], lig_block_chains[i], i, d))

    return epitope, dists
    

if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description='get interface')
    parser.add_argument('--pdb', type=str, required=True, help='Path to the complex pdb')
    parser.add_argument('--target_chains', type=str, nargs='+', required=True, help='Specify target chain ids')
    parser.add_argument('--ligand_chains', type=str, nargs='+', required=True, help='Specify ligand chain ids')
    parser.add_argument('--pocket_th', type=int, default=10.0, help='CB distance threshold for defining the binding site')
    parser.add_argument('--out', type=str, default=None, help='Save epitope information to json file if specified')
    args = parser.parse_args()
    epitope, dists = get_interface(args.pdb, args.target_chains, args.ligand_chains, args.pocket_th)
    para_res = {}
    for _, chain_name, i, d in dists:
        key = f'{chain_name}-{i}'
        para_res[key] = 1
    print(f'REMARK: {len(epitope)} residues in the binding site on the target protein, with {len(para_res)} residues in ligand:')
    print(f' \tchain\tresidue id\ttype\tchain\tresidue id\ttype\tdistance')
    for i, (e, p) in enumerate(zip(epitope, dists)):
        e_res, e_chain_name, _ = e
        p_res, p_chain_name, _, d = p
        print(f'{i+1}\t{e_chain_name}\t{e_res.id}\t{e_res.abrv}\t' + \
              f'{p_chain_name}\t{p_res.id}\t{p_res.abrv}\t{round(d, 3)}')

    if args.out:
        data = []
        for e in epitope:
            res, chain_name, _ = e
            data.append((chain_name, res.id))
        with open(args.out, 'w') as fout:
            json.dump(data, fout)