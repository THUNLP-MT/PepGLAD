#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple
from functools import wraps

import torch


OOMReturn = namedtuple('OOMReturn', ['fake_loss'])


def oom_decorator(forward):
    @wraps(forward)

    def deco_func(self, *args, **kwargs):
        try:
            output = forward(self, *args, **kwargs)
            return output
        except RuntimeError as e:
            if 'out of memory' in str(e):
                output = sum([p.norm() for p in self.parameters() if p.dtype == torch.float]) * 0.0
                return OOMReturn(output)
            else:
                raise e
    
    return deco_func


def safe_backward(loss, model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        loss.backward() # regrettedly, we cannot handle backward oom in distributed training
        return True
    
    try:
        loss.backward()
        return True
    except RuntimeError as e:
        if 'out of memory' in str(e):
            fake_loss = sum([p.norm() for p in model.parameters() if p.dtype == torch.float]) * 0.0
            fake_loss.backward()
            torch.cuda.empty_cache()
            return False
        else:
            raise e