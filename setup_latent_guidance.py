#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import yaml
import argparse
from tqdm import tqdm

import torch

from generate import get_best_ckpt, to_device
from data import create_dataloader, create_dataset


def main(args):
    config = yaml.safe_load(open(args.config, 'r'))
    # load model
    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    print(f'Using checkpoint {b_ckpt}')
    model = torch.load(b_ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()
    
    # load data
    _, _, test_set = create_dataset(config['dataset'])
    test_loader = create_dataloader(test_set, config['dataloader'])

    all_dists = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = to_device(batch, device)
            H, Z, _, _ = model.autoencoder.encode(
                batch['X'], batch['S'], batch['mask'], batch['position_ids'],
                batch['lengths'], batch['atom_mask'], no_randomness=True
            )
            pos = batch['position_ids'][batch['mask']]
            Z = Z.squeeze(1)
            dists = torch.norm(Z[1:] - Z[:-1], dim=-1)  # [N]
            pos_dist = pos[1:] - pos[:-1]
            dists = dists[pos_dist == 1]
            all_dists.append(dists)
    all_dists = torch.cat(all_dists, dim=0)
    mean, std = torch.mean(all_dists), torch.std(all_dists)
    print(mean, std)
    model.set_consec_dist(mean.item(), std.item())
    torch.save(model, b_ckpt)

def parse():
    parser = argparse.ArgumentParser(description='Calculate distance between consecutive latent points')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())