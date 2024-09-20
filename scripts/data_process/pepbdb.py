#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
from tqdm import tqdm

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO

from data.mmap_dataset import create_mmap
from data.format import VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from data.converter.blocks_interface import blocks_interface, blocks_cb_interface
from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Process PepBDB dataset')
    parser.add_argument('--index', type=str, default=None, help='Index file of the dataset')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    parser.add_argument('--pocket_th', type=float, default=10.0,
                        help='Threshold for determining pocket')
    return parser.parse_args()


def process_iterator(items, pdb_dir, pdb_out_dir, pocket_th):
    if not os.path.exists(pdb_out_dir):
        os.makedirs(pdb_out_dir)

    for cnt, pdb_id in enumerate(items):
        summary = items[pdb_id]
        rec_chain, lig_chain = summary['rec_chain'], summary['pep_chain']
        non_standard = 0
        try:
            rec_blocks = pdb_to_list_blocks(os.path.join(pdb_dir, pdb_id, 'receptor.pdb'), selected_chains=[rec_chain])[0]
            lig_blocks = pdb_to_list_blocks(os.path.join(pdb_dir, pdb_id, 'peptide.pdb'), selected_chains=[lig_chain])[0]
        except (KeyError, FileNotFoundError):
            continue
        _, (_, pep_if_idx) = blocks_interface(rec_blocks, lig_blocks, 6.0) # 6A for atomic interaction
        # if len(pep_if_idx) / len(lig_blocks) < 0.3:  # too less contacts
        #     continue
        if len(pep_if_idx) == 0:
            continue
        try:
            _, (pocket_idx, _) = blocks_cb_interface(rec_blocks, lig_blocks, pocket_th)  # 10A for pocket size based on CB
        except KeyError:
            print_log(f'{pdb_id} missing backbone atoms')
            continue # missing both CB and backbone atoms
        rec_num_units = sum([len(block) for block in rec_blocks])
        lig_num_units = sum([len(block) for block in lig_blocks])

        data = ([block.to_tuple() for block in rec_blocks], [block.to_tuple() for block in lig_blocks])
        rec_seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in rec_blocks])
        lig_seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in lig_blocks])
        
        # if '?' in [rec_seq[i] for i in pocket_idx] or '?' in lig_seq:
        if '?' in lig_seq:
            non_standard = 1  # has non-standard amino acids
        
        try:
            list_blocks_to_pdb(
                [rec_blocks, lig_blocks],
                [rec_chain, lig_chain],
                os.path.join(pdb_out_dir, pdb_id + '.pdb')
            )
        except Exception:
            # things like XE1 in 4cin_C, unknown atom
            continue

        yield pdb_id, data, [
            len(rec_blocks), len(lig_blocks), rec_num_units, lig_num_units,
            rec_chain, lig_chain, rec_seq, lig_seq, non_standard,
            ','.join([str(idx) for idx in pocket_idx]),
            ], cnt


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get index file
    with open(args.index, 'r') as fin:
        lines = fin.readlines()
    indexes = {}
    for line in lines:
        line = re.split(r'\s+', line.strip())
        if line[-1] != 'prot':
            continue
        pdb_id = line[0]
        indexes[pdb_id + '_' + line[1]] = {
            'rec_chain': line[4],
            'pep_chain': line[1]
        }
    print_log(f'Total {len(indexes)} entries')
    # 2. process pdb files into our format (mmap)
    create_mmap(
        process_iterator(
            indexes,
            os.path.join(os.path.dirname(args.index), 'pepbdb'),
            os.path.join(args.out_dir, 'pdbs'),
            args.pocket_th
        ),
        args.out_dir, len(indexes))
    
    print_log('Finished!')

    return


if __name__ == '__main__':
    main(parse())