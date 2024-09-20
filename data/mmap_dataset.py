#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import io
import gzip
import json
import mmap
from typing import Optional
from tqdm import tqdm

import torch


def compress(x):
    serialized_x = json.dumps(x).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as f:
        f.write(serialized_x)
    compressed = buf.getvalue()
    return compressed


def decompress(compressed_x):
    buf = io.BytesIO(compressed_x)
    with gzip.GzipFile(fileobj=buf, mode="rb") as f:
        serialized_x = f.read().decode()
    x = json.loads(serialized_x)
    return x


def _find_measure_unit(num_bytes):
    size, measure_unit = num_bytes, 'Bytes'
    for unit in ['KB', 'MB', 'GB']:
        if size > 1000:
            size /= 1024
            measure_unit = unit
        else:
            break
    return size, measure_unit


def create_mmap(iterator, out_dir, total_len=None, commit_batch=10000):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_file_path = os.path.join(out_dir, 'data.bin')
    data_file = open(data_file_path, 'wb')
    index_file = open(os.path.join(out_dir, 'index.txt'), 'w')

    i, offset, n_finished = 0, 0, 0
    progress_bar = tqdm(iterator, total=total_len)
    for _id, x, properties, entry_idx in iterator:
        progress_bar.set_description(f'Processing {_id}')
        compressed_x = compress(x)
        bin_length = data_file.write(compressed_x)
        properties = '\t'.join([str(prop) for prop in properties])
        index_file.write(f'{_id}\t{offset}\t{offset + bin_length}\t{properties}\n') # tuple of (_id, start, end), data slice is [start, end)
        offset += bin_length
        i += 1

        if entry_idx > n_finished:
            progress_bar.update(entry_idx - n_finished)
            n_finished = entry_idx
            if total_len is not None:
                expected_size = os.fstat(data_file.fileno()).st_size / n_finished * total_len
                expected_size, measure_unit = _find_measure_unit(expected_size)
                progress_bar.set_postfix({f'{i} saved; Estimated total size ({measure_unit})': expected_size})

        if i % commit_batch == 0:
            data_file.flush()  # save from memory to disk
            index_file.flush()

        
    data_file.close()
    index_file.close()


class MMAPDataset(torch.utils.data.Dataset):
    
    def __init__(self, mmap_dir: str, specify_data: Optional[str]=None, specify_index: Optional[str]=None) -> None:
        super().__init__()

        self._indexes = []
        self._properties = []
        _index_path = os.path.join(mmap_dir, 'index.txt') if specify_index is None else specify_index
        with open(_index_path, 'r') as f:
            for line in f.readlines():
                messages = line.strip().split('\t')
                _id, start, end = messages[:3]
                _property = messages[3:]
                self._indexes.append((_id, int(start), int(end)))
                self._properties.append(_property)
        _data_path = os.path.join(mmap_dir, 'data.bin') if specify_data is None else specify_data
        self._data_file = open(_data_path, 'rb')
        self._mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __del__(self):
        self._mmap.close()
        self._data_file.close()

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        
        _, start, end = self._indexes[idx]
        data = decompress(self._mmap[start:end])

        return data