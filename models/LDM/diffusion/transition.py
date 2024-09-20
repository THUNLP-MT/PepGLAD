import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_transition(_type, num_steps, opt):
    if _type == 'Diffusion':
        return ContinuousTransition(num_steps, opt)
    elif _type == 'FlowMatching':
        return FlowMatchingTransition(num_steps, opt)
    else:
        raise NotImplementedError(f'transition type {_type} not implemented')


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


class ContinuousTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def get_timestamp(self, t):
        # use beta as timestamp
        return self.var_sched.betas[t]

    def add_noise(self, p_0, mask_generate, batch_ids, t):
        """
        Args:
            p_0: [N, ...]
            mask_generate: [N]
            batch_ids: [N]
            t: [batch_size]
        """
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        alpha_bar = self.var_sched.alpha_bars[t] # [batch_size]
        alpha_bar = alpha_bar[batch_ids]  # [N]

        c0 = torch.sqrt(alpha_bar).view(*expand_shape)
        c1 = torch.sqrt(1 - alpha_bar).view(*expand_shape)

        e_rand = torch.randn_like(p_0)  # [N, 14, 3]
        supervise_e_rand = e_rand.clone()
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate.expand_as(p_0), p_noisy, p_0)

        return p_noisy, supervise_e_rand

    def denoise(self, p_t, eps_p, mask_generate, batch_ids, t, guidance=None, guidance_weight=1.0):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )[batch_ids]
        alpha_bar = self.var_sched.alpha_bars[t][batch_ids]
        sigma = self.var_sched.sigmas[t][batch_ids].view(*expand_shape)

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(*expand_shape)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(*expand_shape)

        z = torch.where(
            (t > 1).view(*expand_shape).expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        if guidance is not None:
            eps_p = eps_p - torch.sqrt(1 - alpha_bar).view(*expand_shape) * guidance

        # if guidance is not None:
        #     p_next = c0 * (p_t - c1 * eps_p) + sigma * z + sigma * sigma * guidance_weight * guidance
        # else:
        #     p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate.expand_as(p_t), p_next, p_t)
        return p_next


# TODO: flow matching (uniform or OT), not done yet
class FlowMatchingTransition(nn.Module):

    def __init__(self, num_steps, opt={}):
        super().__init__()
        self.num_steps = num_steps
        # TODO: number of steps T or T + 1
        c1 = torch.arange(0, num_steps + 1).float() / num_steps
        c0 = 1 - c1
        self.register_buffer('c0', c0)
        self.register_buffer('c1', c1)

    def get_timestamp(self, t):
        # use c1 as timestamp
        return self.c1[t]

    def add_noise(self, p_0, mask_generate, batch_ids, t):
        """
        Args:
            p_0: [N, ...]
            mask_generate: [N]
            batch_ids: [N]
            t: [batch_size]
        """
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        c0 = self.c0[t][batch_ids].view(*expand_shape)
        c1 = self.c1[t][batch_ids].view(*expand_shape)

        e_rand = torch.randn_like(p_0)  # [N, 14, 3]
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate.expand_as(p_0), p_noisy, p_0)

        return p_noisy, (e_rand - p_0)

    def denoise(self, p_t, eps_p, mask_generate, batch_ids, t):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        p_next = p_t - eps_p / self.num_steps
        p_next = torch.where(mask_generate.expand_as(p_t), p_next, p_t)
        return p_next
