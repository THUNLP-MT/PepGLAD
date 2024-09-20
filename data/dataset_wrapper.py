from typing import Callable
from tqdm import tqdm
from math import log

import numpy as np
import torch
import sympy

from utils import register as R


class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, *datasets, collate_fn: Callable=None) -> None:
        super().__init__()
        self.datasets = datasets
        self.cum_len = []
        self.total_len = 0
        for dataset in datasets:
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        self.collate_fn = self.datasets[0].collate_fn if collate_fn is None else collate_fn
        if hasattr(datasets[0], '_lengths'):
            self._lengths = []
            for dataset in datasets:
                self._lengths.extend(dataset._lengths)
    
    def update_epoch(self):
        for dataset in self.datasets:
            if hasattr(dataset, 'update_epoch'):
                dataset.update_epoch()

    def get_len(self, idx):
        return self._lengths[idx]

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i].__getitem__(idx - last_cum_len)
            last_cum_len = cum_len
        return None # this is not possible
    

@R.register('DynamicBatchWrapper')
class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, complexity, ubound_per_batch) -> None:
        super().__init__()
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset))]
        self.complexity = complexity
        self.eval_func = sympy.lambdify('n', sympy.simplify(complexity))
        self.ubound_per_batch = ubound_per_batch
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError(f"'DynamicBatchWrapper'(or '{type(self.dataset)}') object has no attribute '{attr}'")

    def update_epoch(self):
        if hasattr(self.dataset, 'update_epoch'):
            self.dataset.update_epoch()
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):

        np.random.shuffle(self.indexes)
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        cur_complexity = 0
        batch = []

        for i in tqdm(self.indexes):
            item_len = self.eval_func(self.dataset.get_len(i))
            if item_len > self.ubound_per_batch:
                continue
            cur_complexity += item_len
            if cur_complexity > self.ubound_per_batch:
                self.batch_indexes.append(batch)
                batch = []
                cur_complexity = item_len
            batch.append(i)
        self.batch_indexes.append(batch)

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes = self.batch_indexes + last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)