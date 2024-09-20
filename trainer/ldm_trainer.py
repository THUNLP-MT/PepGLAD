#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import pi, cos

import torch
from torch_scatter import scatter_mean

from .abs_trainer import Trainer
from utils import register as R


@R.register('LDMTrainer')
class LDMTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict, criterion: str='AAR'):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.max_step = self.config.max_epoch * len(self.train_loader)
        self.criterion = criterion
        assert criterion in ['AAR', 'RMSD', 'Loss'], f'Criterion {criterion} not implemented'
        self.rng_state = None

    ########## Override start ##########

    def train_step(self, batch, batch_idx):
        results = self.model(**batch)
        if self.is_oom_return(results):
            return results
        loss, loss_dict = results

        self.log('Overall/Loss/Train', loss, batch_idx, val=False)

        if 'H' in loss_dict:
            self.log('Seq/Loss_H/Train', loss_dict['H'], batch_idx, val=False)

        if 'X' in loss_dict:
            self.log('Struct/Loss_X/Train', loss_dict['X'], batch_idx, val=False)

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log('lr', lr, batch_idx, val=False)

        return loss

    def _valid_epoch_begin(self, device):
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(12) # each validation epoch uses the same initial state
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        return super()._valid_epoch_end(device)

    def valid_step(self, batch, batch_idx):
        loss, loss_dict = self.model(**batch)
        self.log('Overall/Loss/Validation', loss, batch_idx, val=True)
        if 'H' in loss_dict: self.log('Seq/Loss_H/Validation', loss_dict['H'], batch_idx, val=True)
        if 'X' in loss_dict: self.log('Struct/Loss_X/Validation', loss_dict['X'], batch_idx, val=True)
        # disable sidechain optimization as it may stuck for early validations where the model is still weak
        if self.local_rank != -1:  # ddp
            sample_X, sample_S, _ = self.model.module.sample(**batch, return_tensor=True, optimize_sidechain=False)
        else:
            sample_X, sample_S, _ = self.model.sample(**batch, return_tensor=True, optimize_sidechain=False)
        mask_generate = batch['mask']
        # batch ids
        batch_ids = torch.zeros_like(mask_generate).long()
        batch_ids[torch.cumsum(batch['lengths'], dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        batch_ids = batch_ids[mask_generate]

        if sample_S is not None:
            # aar
            aar = (batch['S'][mask_generate] == sample_S).float()
            aar = torch.mean(scatter_mean(aar, batch_ids, dim=-1))
            self.log('Seq/AAR/Validation', aar, batch_idx, val=True)

        # ca rmsd
        if sample_X is not None:
            atom_mask = batch['atom_mask'][mask_generate][:, 1]
            rmsd = ((batch['X'][mask_generate][:, 1][atom_mask] - sample_X[:, 1][atom_mask]) ** 2).sum(-1)  # [Ntgt]
            rmsd = torch.sqrt(scatter_mean(rmsd, batch_ids[atom_mask], dim=-1))  # [bs]
            rmsd = torch.mean(rmsd)

            self.log('Struct/CA_RMSD/Validation', rmsd, batch_idx, val=True)

        if self.criterion == 'AAR':
            return aar.detach()
        elif self.criterion == 'RMSD':
            return rmsd.detach()
        elif self.criterion == 'Loss':
            return loss.detach()
        else:
            raise NotImplementedError(f'Criterion {self.criterion} not implemented')

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    ########## Override end ##########