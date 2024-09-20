#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F

from data.format import VOCAB
from utils.nn_utils import graph_to_batch


@torch.no_grad()
def continuous_bool(x, k=1000):
    return (x > 0).float()


def _consec_dist_loss(gen_X, gen_X_mask, lb, ub, eps=1e-6):
    consec_dist = torch.norm(gen_X[..., 1:, :] - gen_X[..., :-1, :], dim=-1) # [bs, max_L - 1]
    consec_lb_loss = lb - consec_dist  # [bs, max_L - 1]
    consec_ub_loss = consec_dist - ub  # [bs, max_L - 1]

    consec_lb_invalid = (consec_dist < lb) & gen_X_mask[..., 1:]
    consec_ub_invalid = (consec_dist > ub) & gen_X_mask[..., 1:]
    consec_loss = torch.where(consec_lb_invalid, consec_lb_loss, torch.zeros_like(consec_lb_loss))
    consec_loss = torch.where(consec_ub_invalid, consec_ub_loss, consec_loss)

    consec_loss = consec_loss.sum(-1) / (consec_lb_invalid + consec_ub_invalid + eps).sum(-1)
    consec_loss = torch.sum(consec_loss) # consistent loss scale across different batch size
    return consec_loss


def _inner_clash_loss(gen_X, gen_X_mask, mean, eps=1e-6):
    dist = torch.norm(gen_X[..., :, None, :] - gen_X[..., None, :, :], dim=-1) # [bs, max_L, max_L]
    dist_mask = gen_X_mask[..., :, None] & gen_X_mask[..., None, :] # [bs, max_L, max_L]
    pos = torch.cumsum(torch.ones_like(gen_X_mask, dtype=torch.long), dim=-1) # [bs, max_L]
    non_consec_mask = torch.abs(pos[..., :, None] - pos[..., None, :]) > 1  # [bs, max_L, max_L]

    clash_loss = mean - dist
    clash_loss_mask = (clash_loss > 0) & dist_mask & non_consec_mask # [bs, max_L, max_L]
    clash_loss = torch.where(clash_loss_mask, clash_loss, torch.zeros_like(clash_loss))

    clash_loss = clash_loss.sum(-1).sum(-1) / (clash_loss_mask.sum(-1).sum(-1) + eps)
    clash_loss = torch.sum(clash_loss)  # consistent loss scale across different residue number and batch size
    return clash_loss


def _outer_clash_loss(ctx_X, ctx_X_mask, gen_X, gen_X_mask, mean, eps=1e-6):
    dist = torch.norm(gen_X[..., :, None, :] - ctx_X[..., None, :, :], dim=-1)  # [bs, max_gen_L, max_ctx_L]
    dist_mask = gen_X_mask[..., :, None] & ctx_X_mask[..., None, :] # [bs, max_gen_L, max_ctx_L]
    clash_loss = mean - dist  # [bs, max_gen_L, max_ctx_L]
    clash_loss_mask = (clash_loss > 0) & dist_mask  # [bs, max_gen_L, max_ctx_L]
    clash_loss = torch.where(clash_loss_mask, clash_loss, torch.zeros_like(clash_loss))

    clash_loss = clash_loss.sum(-1).sum(-1) / (clash_loss_mask.sum(-1).sum(-1) + eps)
    clash_loss = torch.sum(clash_loss)  # consistent loss scale across different residue number and batch size
    return clash_loss


def dist_energy(X, mask_generate, batch_ids, mean, std, tolerance=3, **kwargs):
    mean, std = round(mean, 4), round(std, 4)
    lb, ub = mean - tolerance * std, mean + tolerance * std

    X = X.clone() # [N, 3]

    ctx_X, ctx_batch_ids = X[~mask_generate], batch_ids[~mask_generate]
    gen_X, gen_batch_ids = X[mask_generate], batch_ids[mask_generate]
    ctx_X = ctx_X[:, VOCAB.ca_channel_idx] # CA (alpha carbon)
    gen_X = gen_X[:, 0] # latent one

    # to batch representation
    ctx_X, ctx_X_mask = graph_to_batch(ctx_X, ctx_batch_ids, mask_is_pad=False) # [bs, max_ctx_L, 3]
    gen_X, gen_X_mask = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False) # [bs, max_gen_L, 3]

    # consecutive
    consec_loss = _consec_dist_loss(gen_X, gen_X_mask, lb, ub)

    # inner clash
    inner_clash_loss = _inner_clash_loss(gen_X, gen_X_mask, mean)

    # outer clash
    outer_clash_loss = _outer_clash_loss(ctx_X, ctx_X_mask, gen_X, gen_X_mask, mean)

    return consec_loss + inner_clash_loss + outer_clash_loss
