#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np


def blocks_to_coords(blocks):
    max_n_unit = 0
    coords, masks = [], []
    for block in blocks:
        coords.append([unit.get_coord() for unit in block.units])
        max_n_unit = max(max_n_unit, len(coords[-1]))
        masks.append([1 for _ in coords[-1]])
    
    for i in range(len(coords)):
        num_pad =  max_n_unit - len(coords[i])
        coords[i] = coords[i] + [[0, 0, 0] for _ in range(num_pad)]
        masks[i] = masks[i] + [0 for _ in range(num_pad)]
    
    return np.array(coords), np.array(masks).astype('bool')  # [N, M, 3], [N, M], M == max_n_unit, in mask 0 is for padding


def dist_matrix_from_coords(coords1, masks1, coords2, masks2):
    dist = np.linalg.norm(coords1[:, None] - coords2[None, :], axis=-1)  # [N1, N2, M]
    dist = dist + np.logical_not(masks1[:, None] * masks2[None, :]) * 1e6  # [N1, N2, M]
    dist = np.min(dist, axis=-1)  # [N1, N2]
    return dist


def dist_matrix_from_blocks(blocks1, blocks2):
    blocks_coord, blocks_mask = blocks_to_coords(blocks1 + blocks2)
    blocks1_coord, blocks1_mask = blocks_coord[:len(blocks1)], blocks_mask[:len(blocks1)]
    blocks2_coord, blocks2_mask = blocks_coord[len(blocks1):], blocks_mask[len(blocks1):]
    dist = dist_matrix_from_coords(blocks1_coord, blocks1_mask, blocks2_coord, blocks2_mask)
    return dist


def blocks_interface(blocks1, blocks2, dist_th):
    dist = dist_matrix_from_blocks(blocks1, blocks2)
    
    on_interface = dist < dist_th
    indexes1 = np.nonzero(on_interface.sum(axis=1) > 0)[0]
    indexes2 = np.nonzero(on_interface.sum(axis=0) > 0)[0]

    blocks1 = [blocks1[i] for i in indexes1]
    blocks2 = [blocks2[i] for i in indexes2]

    return (blocks1, blocks2), (indexes1, indexes2)


def add_cb(input_array):
    #from protein mpnn
    #The virtual Cβ coordinates were calculated using ideal angle and bond length definitions: b = Cα - N, c = C - Cα, a = cross(b, c), Cβ = -0.58273431*a + 0.56802827*b - 0.54067466*c + Cα.
    N,CA,C,O = input_array
    b = CA - N
    c = C - CA
    a = np.cross(b,c)
    CB = np.around(-0.58273431*a + 0.56802827*b - 0.54067466*c + CA,3)
    return CB #np.array([N,CA,C,CB,O])


def blocks_to_cb_coords(blocks):
    cb_coords = []
    for block in blocks:
         try:
              cb_coords.append(block.get_unit_by_name('CB').get_coord())
         except KeyError:
              tmp_coord = np.array([
                   block.get_unit_by_name('N').get_coord(),
                   block.get_unit_by_name('CA').get_coord(),
                   block.get_unit_by_name('C').get_coord(),
                   block.get_unit_by_name('O').get_coord()
              ])
              cb_coords.append(add_cb(tmp_coord))
    return np.array(cb_coords)


def blocks_cb_interface(blocks1, blocks2, dist_th=8.0):
    cb_coords1 = blocks_to_cb_coords(blocks1)
    cb_coords2 = blocks_to_cb_coords(blocks2)
    dist = np.linalg.norm(cb_coords1[:, None] - cb_coords2[None, :], axis=-1)  # [N1, N2]
    
    on_interface = dist < dist_th
    indexes1 = np.nonzero(on_interface.sum(axis=1) > 0)[0]
    indexes2 = np.nonzero(on_interface.sum(axis=0) > 0)[0]

    blocks1 = [blocks1[i] for i in indexes1]
    blocks2 = [blocks2[i] for i in indexes2]

    return (blocks1, blocks2), (indexes1, indexes2)