#!/usr/bin/env python3
import torch
from torch import nn
from contextlib import contextmanager
from collections import OrderedDict

class Sequential(nn.Sequential):
    @contextmanager
    def ondev(self, device):
        try:
            self._rmtodev(device)
            yield self
        finally:
            self._rmtodev(torch.device('cpu'))

    def _rmtodev(self, dev):
        rmparams = self.parameters()
        self.to(dev)
        for param in rmparams:
            del param

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return __class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

class Parallel(nn.Module):
    pass

class Concat(nn.Module):
    pass

class Multiplexer(nn.Module):
    pass

class BatchView(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self._shape = shape

    def forward(self, x):
        ishape = x.shape
        return x.view(ishape[0], *self._shape)

