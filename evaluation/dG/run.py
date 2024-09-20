#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import statistics

import ray

from utils.logger import print_log

from .base import TaskScanner, run_pyrosetta

# @ray.remote(num_gpus=1/8, num_cpus=1)
# def run_openmm_remote(task):
#     return run_openmm(task)


@ray.remote(num_cpus=1)
def run_pyrosetta_remote(task):
    return run_pyrosetta(task)


@ray.remote
def pipeline_pyrosetta(task):
    funcs = [
        run_pyrosetta_remote,
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)


def parse():
    parser = argparse.ArgumentParser(description='calculating dG using pyrosetta')
    parser.add_argument('--results', type=str, required=True, help='Path to the summary of the results (.jsonl)')
    parser.add_argument('--n_sample', type=int, default=float('inf'), help='Maximum number of samples for calculation')
    parser.add_argument('--rfdiff_relax', action='store_true', help='Use rfdiff fastrelax')
    parser.add_argument('--out_path', type=str, default=None, help='Output path, default dG_report.jsonl under the same directory as results')
    return parser.parse_args()


def main(args):
    # output summary
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.results), 'dG_report.jsonl')
    results = {}

    # parallel
    ray.init()
    scanner = TaskScanner(args.results, args.n_sample, args.rfdiff_relax)
    if args.results.endswith('txt'):
        tasks = scanner.scan_dataset()
    else:
        tasks = scanner.scan()
    futures = [pipeline_pyrosetta.remote(t) for t in tasks]
    if len(futures) > 0:
        print_log(f'Submitted {len(futures)} tasks.')
    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            done_task = ray.get(done_id)
            print_log(f'Remaining {len(futures)}. Finished {done_task.current_path}, dG {done_task.dG}')
            _id, number = done_task.info['id'], done_task.info['number']
            if _id not in results:
                results[_id] = {
                    'min': float('inf'),
                    'all': {}
                }
            results[_id]['all'][number] = done_task.dG
            results[_id]['min'] = min(results[_id]['min'], done_task.dG)
    
    # write results
    for _id in results:
        success = 0
        for n in results[_id]['all']:
            if results[_id]['all'][n] < 0:
                success += 1
        results[_id]['success rate'] = success / len(results[_id]['all'])
    json.dump(results, open(args.out_path, 'w'), indent=2)

    # show results
    vals = [results[_id]['min'] for _id in results]
    print(f'median: {statistics.median(vals)}, mean: {sum(vals) / len(vals)}')
    success = [results[_id]['success rate'] for _id in results]
    print(f'mean success rate: {sum(success) / len(success)}')


if __name__ == '__main__':
    import random
    random.seed(12)
    main(parse())
