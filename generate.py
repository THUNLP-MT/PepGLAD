#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool

import yaml
import torch
from torch.utils.data import DataLoader

import models
from utils.config_utils import overwrite_values
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from data.format import VOCAB, Atom
from data import create_dataloader, create_dataset
from utils.logger import print_log
from utils.random_seed import setup_seed
from utils.const import sidechain_atoms


def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = []
    for l in ls:
        k,v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k,v))

    # ckpts = sorted(ckpts, key=lambda x:x[0])
    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def clamp_coord(coord):
    # some models (e.g. diffab) will output very large coordinates (absolute value >1000) which will corrupt the pdb file
    new_coord = []
    for val in coord:
        if abs(val) >= 1000:
            val = 0
        new_coord.append(val)
    return new_coord


def overwrite_blocks(blocks, seq=None, X=None):
    if seq is not None:
        assert len(blocks) == len(seq), f'{len(blocks)} {len(seq)}'
    new_blocks = []
    for i, block in enumerate(blocks):
        block = deepcopy(block)
        if seq is None:
            abrv = block.abrv
        else:
            abrv = VOCAB.symbol_to_abrv(seq[i])
            if block.abrv != abrv:
                if X is None:
                    block.units = [atom for atom in block.units if atom.name in VOCAB.backbone_atoms]
        if X is not None:
            coords = X[i]
            atoms = VOCAB.backbone_atoms + sidechain_atoms[VOCAB.abrv_to_symbol(abrv)]
            block.units = [
                Atom(atom_name, clamp_coord(coord), atom_name[0]) for atom_name, coord in zip(atoms, coords)
            ]
        block.abrv = abrv
        new_blocks.append(block)
    return new_blocks


def generate_wrapper(model, sample_opt={}):
    if isinstance(model, models.AutoEncoder):
        def wrapper(batch):
            X, S, ppls = model.test(batch['X'], batch['S'], batch['mask'], batch['position_ids'], batch['lengths'], batch['atom_mask'])
            return X, S, ppls
    elif isinstance(model, models.LDMPepDesign):
        def wrapper(batch):
            X, S, ppls = model.sample(batch['X'], batch['S'], batch['mask'], batch['position_ids'], batch['lengths'], batch['atom_mask'],
                                      L=batch['L'] if 'L' in batch else None, sample_opt=sample_opt)
            return X, S, ppls
    else:
        raise NotImplementedError(f'Wrapper for {type(model)} not implemented')
    return wrapper


def save_data(
        _id, n,
        x_pkl_file, s_pkl_file, pmetric_pkl_file,
        ref_pdb, rec_chain, lig_chain, ref_save_dir, cand_save_dir,
        seq_only, struct_only, backbone_only
    ):

    X, S, pmetric = pkl.load(open(x_pkl_file, 'rb')), pkl.load(open(s_pkl_file, 'rb')), pkl.load(open(pmetric_pkl_file, 'rb'))
    os.remove(x_pkl_file), os.remove(s_pkl_file), os.remove(pmetric_pkl_file)
    if seq_only:
        X = None
    elif struct_only:
        S = None
    rec_blocks, lig_blocks = pdb_to_list_blocks(ref_pdb, selected_chains=[rec_chain, lig_chain])
    ref_pdb = os.path.join(ref_save_dir, _id + "_ref.pdb")
    list_blocks_to_pdb([rec_blocks, lig_blocks], [rec_chain, lig_chain], ref_pdb)
    # os.system(f'cp {ref_pdb} {os.path.join(ref_save_dir, _id + "_ref.pdb")}')
    ref_seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in lig_blocks])
    lig_blocks = overwrite_blocks(lig_blocks, S, X)
    gen_seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in lig_blocks])
    save_dir = os.path.join(cand_save_dir, _id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gen_pdb = os.path.join(save_dir, _id + f'_gen_{n}.pdb')
    list_blocks_to_pdb([rec_blocks, lig_blocks], [rec_chain, lig_chain], gen_pdb)

    return {
            'id': _id,
            'number': n,
            'gen_pdb': gen_pdb,
            'ref_pdb': ref_pdb,
            'pmetric': pmetric,
            'rec_chain': rec_chain,
            'lig_chain': lig_chain,
            'ref_seq': ref_seq,
            'gen_seq': gen_seq,
            'seq_only': seq_only,
            'struct_only': struct_only,
            'backbone_only': backbone_only
    }


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    struct_only = config.get('struct_only', False)
    seq_only = config.get('seq_only', False)
    assert not (seq_only and struct_only)
    backbone_only = config.get('backbone_only', False)
    # load model
    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    ckpt_dir = os.path.split(os.path.split(b_ckpt)[0])[0]
    print(f'Using checkpoint {b_ckpt}')
    model = torch.load(b_ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load data
    _, _, test_set = create_dataset(config['dataset'])
    test_loader = create_dataloader(test_set, config['dataloader'])
    
    # save path
    if args.save_dir is None:
        save_dir = os.path.join(ckpt_dir, 'results')
    else:
        save_dir = args.save_dir
    ref_save_dir = os.path.join(save_dir, 'references')
    cand_save_dir = os.path.join(save_dir, 'candidates')
    for directory in [ref_save_dir, cand_save_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    

    fout = open(os.path.join(save_dir, 'results.jsonl'), 'w')
    item_idx = 0

    # multiprocessing
    pool = Pool(args.n_cpu)

    n_samples = config.get('n_samples', 1)

    pbar = tqdm(total=n_samples * len(test_loader))
    for n in range(n_samples):
        item_idx = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = to_device(batch, device)
                batch_X, batch_S, batch_pmetric = generate_wrapper(model, deepcopy(config.get('sample_opt', {})))(batch)

                # parallel
                inputs = []
                for X, S, pmetric in zip(batch_X, batch_S, batch_pmetric):
                    _id, ref_pdb, rec_chain, lig_chain = test_set.get_summary(item_idx)
                    # save temporary pickle file
                    x_pkl_file = os.path.join(save_dir, _id + f'_gen_{n}_X.pkl')
                    pkl.dump(X, open(x_pkl_file, 'wb'))
                    s_pkl_file = os.path.join(save_dir, _id + f'_gen_{n}_S.pkl')
                    pkl.dump(S, open(s_pkl_file, 'wb'))
                    pmetric_pkl_file = os.path.join(save_dir, _id + f'_gen_{n}_pmetric.pkl')
                    pkl.dump(pmetric, open(pmetric_pkl_file, 'wb'))
                    inputs.append((
                        _id, n,
                        x_pkl_file, s_pkl_file, pmetric_pkl_file,
                        ref_pdb, rec_chain, lig_chain, ref_save_dir, cand_save_dir,
                        seq_only, struct_only, backbone_only
                    ))
                    item_idx += 1
                
                results = pool.starmap(save_data, inputs)
                for result in results:
                    fout.write(json.dumps(result) + '\n')
                
                pbar.update(1)

    fout.close()


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated peptides')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)
