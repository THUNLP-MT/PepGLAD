#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def graph_to_batch(tensor, batch_id, padding_value=0, mask_is_pad=True):
    '''
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    '''
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    batch = torch.ones((bs, max_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) * padding_value
    # generate pad mask: 1 for pad and 0 for data
    pad_mask = torch.zeros((bs, max_n + 1), dtype=torch.long, device=tensor.device)
    pad_mask[(torch.arange(bs, device=tensor.device), lengths)] = 1
    pad_mask = (torch.cumsum(pad_mask, dim=-1)[:, :-1]).bool()
    data_mask = torch.logical_not(pad_mask)
    # fill data
    batch[data_mask] = tensor
    mask = pad_mask if mask_is_pad else data_mask
    return batch, mask


def variadic_arange(size):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_arange

    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range


def variadic_meshgrid(input1, size1, input2, size2):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_meshgrid
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each input,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Parameters:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat_interleave(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat_interleave(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat_interleave(grid_size)
    index1 = torch.div(local_index, local_inner_size, rounding_mode="floor") + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]


@torch.no_grad()
def length_to_batch_id(S, lengths):
    # generate batch id
    batch_id = torch.zeros_like(S)  # [N]
    batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
    batch_id.cumsum_(dim=0)  # [N], item idx in the batch
    return batch_id


def scatter_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    '''
    from https://github.com/rusty1s/pytorch_scatter/issues/48
    WARN: the range between src.max() and src.min() should not be too wide for numerical stability

    reproducible
    '''
    # f_src = src.float()
    # f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    # norm = (f_src - f_min)/(f_max - f_min + eps) + index.float()*(-1)**int(descending)
    # perm = norm.argsort(dim=dim, descending=descending)

    # return src[perm], perm
    src, src_perm = torch.sort(src, dim=dim, descending=descending)
    index = index.take_along_dim(src_perm, dim=dim)
    index, index_perm = torch.sort(index, dim=dim, stable=True)
    src = src.take_along_dim(index_perm, dim=dim)
    perm = src_perm.take_along_dim(index_perm, dim=0)
    return src, perm


def scatter_topk(src: torch.Tensor, index: torch.Tensor, k: int, dim=0, largest=True):
    indices = torch.arange(src.shape[dim], device=src.device)
    src, perm = scatter_sort(src, index, dim, descending=largest)
    index, indices = index[perm], indices[perm]
    mask = torch.ones_like(index).bool()
    mask[k:] = index[k:] != index[:-k]
    return src[mask], indices[mask]


@torch.no_grad()
def knn_edges(all_edges, k_neighbors, X=None, atom_mask=None, given_dist=None):
    '''
    :param all_edges: [2, E], (row, col)
    :param X: [N, n_channel, 3], coordinates
    :param atom_mask: [N, n_channel], 1 for having atom
    :param given_dist: [E], given distance of edges
    IMPORTANT: either given_dist should be given, or both X and atom_mask should be given
    '''
    assert (given_dist is not None) or (X is not None and atom_mask is not None), \
        'either given_dist should be given, or both X and atom_mask should be given'

    # get distance on each edge
    if given_dist is None:
        row, col = all_edges
        dist = torch.norm(X[row][:, :, None, :] - X[col][:, None, :, :], dim=-1)  # [E, n_channel, n_channel]
        dist_mask = atom_mask[row][:, :, None] & atom_mask[col][:, None, :]  # [E, n_channel, n_channel]
        dist = torch.where(dist_mask, dist, torch.ones_like(dist) * float('inf'))  # [E, n_channel, n_channel]
        dist, _ = dist.view(dist.shape[0], -1).min(axis=-1) # [E]
    else:
        dist = given_dist

    # get topk for each node
    _, indices = scatter_topk(dist, row, k=k_neighbors, largest=False)
    edges = torch.stack([all_edges[0][indices], all_edges[1][indices]], dim=0) # [2, k*N]
    return edges  # [2, E]


class EdgeConstructor:
    def __init__(self, cor_idx, col_idx, atom_pos_pad_idx, rec_seg_id) -> None:
        self.cor_idx, self.col_idx = cor_idx, col_idx
        self.atom_pos_pad_idx = atom_pos_pad_idx
        self.rec_seg_id = rec_seg_id

        # buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.row = None
        self.col = None
        self.row_global = None
        self.col_global = None
        self.row_seg = None
        self.col_seg = None
        self.offsets = None
        self.max_n = None
        self.gni2lni = None
        self.not_global_edges = None

    def get_batch_edges(self, batch_id):
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        return (row, col), (offsets, max_n, gni2lni)

    def _prepare(self, S, batch_id, segment_ids) -> None:
        (row, col), (offsets, max_n, gni2lni) = self.get_batch_edges(batch_id)

        # not global edges
        is_global = sequential_or(S == self.cor_idx, S == self.col_idx) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        
        # segment ids
        row_seg, col_seg = segment_ids[row], segment_ids[col]

        # add to buffer
        self.row, self.col = row, col
        self.offsets, self.max_n, self.gni2lni = offsets, max_n, gni2lni
        self.row_global, self.col_global = row_global, col_global
        self.not_global_edges = not_global_edges
        self.row_seg, self.col_seg = row_seg, col_seg

    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: same seg, not global
        select_edges = torch.logical_and(self.row_seg == self.col_seg, self.not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges

    def _construct_outer_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(self.row_seg != self.col_seg, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        outer_edges = _knn_edges(
            X, atom_pos, torch.stack([inter_all_row, inter_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return outer_edges

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(self.row_global, self.col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_seq_edges(self):
        row, col = self.row, self.col
        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph
            self.not_global_edges,  # not global edges (also ensure the edges are in the same segment)
            self.row_seg != self.rec_seg_id  # not epitope
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return seq_adj

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, k_neighbors, atom_pos, segment_ids):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # prepare inputs
        self._prepare(S, batch_id, segment_ids)

        ctx_edges, inter_edges = [], []

        # edges within chains
        inner_edges = self._construct_inner_edges(X, batch_id, k_neighbors, atom_pos)
        # edges between global nodes and normal/global nodes
        global_normal, global_global = self._construct_global_edges()
        # edges on the 1D sequence
        seq_edges = self._construct_seq_edges()

        # # construct context edges
        ctx_edges = torch.cat([inner_edges, global_normal, global_global, seq_edges], dim=1)  # [2, E]

        # construct interaction edges
        inter_edges = self._construct_outer_edges(X, batch_id, k_neighbors, atom_pos)

        self._reset_buffer()
        return ctx_edges, inter_edges


class GMEdgeConstructor(EdgeConstructor):
    '''
    Edge constructor for graph matching (kNN internel edges and all bipartite edges)
    '''
    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: both in ag or ab, not global
        row_is_rec = self.row_seg == self.rec_seg_id
        col_is_rec = self.col_seg == self.rec_seg_id
        select_edges = torch.logical_and(row_is_rec == col_is_rec, self.not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = sequential_and(self.row_global, self.col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_outer_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible inter edges: one in ag and one in ab, not global
        row_is_rec = self.row_seg == self.rec_seg_id
        col_is_rec = self.col_seg == self.rec_seg_id
        select_edges = torch.logical_and(row_is_rec != col_is_rec, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        return torch.stack([inter_all_row, inter_all_col])  # [2, E]


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings