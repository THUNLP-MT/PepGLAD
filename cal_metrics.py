#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import random
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import statistics
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from evaluation import diversity
from evaluation.dockq import dockq
from evaluation.rmsd import compute_rmsd
from utils.random_seed import setup_seed
from evaluation.seq_metric import aar, slide_aar


def _get_ref_pdb(_id, root_dir):
    return os.path.join(root_dir, 'references', f'{_id}_ref.pdb')


def _get_gen_pdb(_id, number, root_dir, use_rosetta):
    suffix = '_rosetta' if use_rosetta else ''
    return os.path.join(root_dir, 'candidates', _id, f'{_id}_gen_{number}{suffix}.pdb')


def cal_metrics(items):
    # all of the items are conditioned on the same binding pocket
    root_dir = items[0]['root_dir']
    ref_pdb, rec_chain, lig_chain = items[0]['ref_pdb'], items[0]['rec_chain'], items[0]['lig_chain']
    ref_pdb = _get_ref_pdb(items[0]['id'], root_dir)
    seq_only, struct_only, backbone_only = items[0]['seq_only'], items[0]['struct_only'], items[0]['backbone_only']

    # prepare
    results = defaultdict(list)
    cand_seqs, cand_ca_xs = [], []
    rec_blocks, ref_pep_blocks = pdb_to_list_blocks(ref_pdb, [rec_chain, lig_chain])
    ref_ca_x, ca_mask = [], []
    for ref_block in ref_pep_blocks:
        if ref_block.has_unit('CA'):
            ca_mask.append(1)
            ref_ca_x.append(ref_block.get_unit_by_name('CA').get_coord())
        else:
            ca_mask.append(0)
            ref_ca_x.append([0, 0, 0])
    ref_ca_x, ca_mask = np.array(ref_ca_x), np.array(ca_mask).astype(bool)

    for item in items:
        if not struct_only:
            cand_seqs.append(item['gen_seq'])
            results['Slide AAR'].append(slide_aar(item['gen_seq'], item['ref_seq'], aar))
        
        # structure metrics
        gen_pdb = _get_gen_pdb(item['id'], item['number'], root_dir, item['rosetta'])
        _, gen_pep_blocks = pdb_to_list_blocks(gen_pdb, [rec_chain, lig_chain])
        assert len(gen_pep_blocks) == len(ref_pep_blocks), f'{item}\t{len(ref_pep_blocks)}\t{len(gen_pep_blocks)}'

        # CA RMSD
        gen_ca_x = np.array([block.get_unit_by_name('CA').get_coord() for block in gen_pep_blocks])
        cand_ca_xs.append(gen_ca_x)
        rmsd = compute_rmsd(ref_ca_x[ca_mask], gen_ca_x[ca_mask], aligned=True)
        results['RMSD(CA)'].append(rmsd)
        if struct_only:
            results['RMSD<=2.0'].append(1 if rmsd <= 2.0 else 0)
            results['RMSD<=5.0'].append(1 if rmsd <= 5.0 else 0)
            results['RMSD<=10.0'].append(1 if rmsd <= 10.0 else 0)


        if backbone_only:
            continue

        # 5. DockQ
        dockq_score = dockq(gen_pdb, ref_pdb, lig_chain)
        results['DockQ'].append(dockq_score)
        if struct_only:
            results['DockQ>=0.23'].append(1 if dockq_score >= 0.23 else 0)
            results['DockQ>=0.49'].append(1 if dockq_score >= 0.49 else 0)
            results['DockQ>=0.80'].append(1 if dockq_score >= 0.80 else 0)

        # Full atom RMSD
        if struct_only:
            gen_all_x, ref_all_x = [], []
            for gen_block, ref_block in zip(gen_pep_blocks, ref_pep_blocks):
                for ref_atom in ref_block:
                    if gen_block.has_unit(ref_atom.name):
                        ref_all_x.append(ref_atom.get_coord())
                        gen_all_x.append(gen_block.get_unit_by_name(ref_atom.name).get_coord())
            results['RMSD(full-atom)'].append(compute_rmsd(
                np.array(gen_all_x), np.array(ref_all_x), aligned=True
            ))
            
    pmets = [item['pmetric'] for item in items]
    indexes = list(range(len(items)))
    # aggregation
    for name in results:
        vals = results[name]
        corr = spearmanr(vals, pmets, nan_policy='omit').statistic
        if np.isnan(corr):
            corr = 0
        aggr_res = {
            'max': max(vals),
            'min': min(vals),
            'mean': sum(vals) / len(vals),
            'random': vals[0],
            'max*': vals[(max if corr > 0 else min)(indexes, key=lambda i: pmets[i])],
            'min*': vals[(min if corr > 0 else max)(indexes, key=lambda i: pmets[i])],
            'pmet_corr': corr,
            'individual': vals,
            'individual_pmet': pmets
        }
        results[name] = aggr_res

    if len(cand_seqs) > 1 and not seq_only:
        seq_div, struct_div, co_div, consistency = diversity.diversity(cand_seqs, np.array(cand_ca_xs))
        results['Sequence Diversity'] = seq_div
        results['Struct Diversity'] = struct_div
        results['Codesign Diversity'] = co_div
        results['Consistency'] = consistency

    return results


def cnt_aa_dist(seqs):
    cnts = {}
    for seq in seqs:
        for aa in seq:
            if aa not in cnts:
                cnts[aa] = 0
            cnts[aa] += 1
    aas = sorted(list(cnts.keys()), key=lambda aa: cnts[aa])
    total = sum(cnts.values())
    for aa in aas:
        print(f'\t{aa}: {cnts[aa] / total}')


def main(args):
    root_dir = os.path.dirname(args.results)
    # load dG filter
    if args.filter_dG is None:
        filter_func = lambda _id, n: True
    else:
        dG_results = json.load(open(args.filter_dG, 'r'))
        filter_func = lambda _id, n: dG_results[_id]['all'][str(n)] < 0
    # load results
    with open(args.results, 'r') as fin:
        lines = fin.read().strip().split('\n')
    id2items = {}
    for line in lines:
        item = json.loads(line)
        _id = item['id']
        if not filter_func(_id, item['number']):
            continue
        if _id not in id2items:
            id2items[_id] = []
        item['root_dir'] = root_dir
        item['rosetta'] = args.rosetta
        id2items[_id].append(item)
    ids = list(id2items.keys())

    if args.filter_dG is not None:
        # delete results with only one sample since it cannot calculate diversity
        del_ids = [_id for _id in ids if len(id2items[_id]) < 2]
        for _id in del_ids:
            print(f'Deleting {_id} since it only has one sample passed the filter')
            del id2items[_id]

    if args.num_workers > 1:
        metrics = process_map(cal_metrics, id2items.values(), max_workers=args.num_workers, chunksize=1)
    else:
        metrics = [cal_metrics(inputs) for inputs in tqdm(id2items.values())]
    
    eval_results_path = os.path.join(os.path.dirname(args.results), 'eval_report.json')
    with open(eval_results_path, 'w') as fout:
        for i, _id in enumerate(id2items):
            metric = deepcopy(metrics[i])
            metric['id'] = _id
            fout.write(json.dumps(metric) + '\n')

    # individual level results
    print('Point-wise evaluation results:')
    for name in metrics[0]:
        vals = [item[name] for item in metrics]
        if isinstance(vals[0], dict):
            if 'RMSD' in name and '<=' not in name:
                aggr = 'min'
            else:
                aggr = 'max'
            aggr_vals = [val[aggr] for val in vals]
            if '>=' in name or '<=' in name:  # percentage
                print(f'{name}: {sum(aggr_vals) / len(aggr_vals)}')
            else:
                if 'RMSD' in name:
                    print(f'{name}(median): {statistics.median(aggr_vals)}') # unbounded, some extreme values will affect the mean but not the median
                else:
                    print(f'{name}(mean): {sum(aggr_vals) / len(aggr_vals)}')
                lowest_i = min([i for i in range(len(aggr_vals))], key=lambda i: aggr_vals[i])
                highest_i = max([i for i in range(len(aggr_vals))], key=lambda i: aggr_vals[i])
                print(f'\tlowest: {aggr_vals[lowest_i]}, id: {ids[lowest_i]}', end='')
                print(f'\thighest: {aggr_vals[highest_i]}, id: {ids[highest_i]}')
        else:
            print(f'{name} (mean): {sum(vals) / len(vals)}')
            lowest_i = min([i for i in range(len(vals))], key=lambda i: vals[i])
            highest_i = max([i for i in range(len(vals))], key=lambda i: vals[i])
            print(f'\tlowest: {vals[lowest_i]}, id: {ids[lowest_i]}')
            print(f'\thighest: {vals[highest_i]}, id: {ids[highest_i]}')


def parse():
    parser = argparse.ArgumentParser(description='calculate metrics')
    parser.add_argument('--results', type=str, required=True, help='Path to test set')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use')
    parser.add_argument('--rosetta', action='store_true', help='Use the rosetta-refined structure')
    parser.add_argument('--filter_dG', type=str, default=None, help='Only calculate results on samples with dG<0')

    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(0)
    main(parse())
