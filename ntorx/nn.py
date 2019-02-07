#!/usr/bin/env python3
import torch
from torch import nn
from contextlib import contextmanager
from collections import OrderedDict

class Sequential(nn.Sequential):
    """
    """
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

class Linear(nn.Module):
    """Abstract Class for linear layers with weights and biases.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = None
        self.bias = None

    @contextmanager
    def with_params(self, weight, bias):
        try:
            old_weight = self.weight
            old_bias = self.bias
            self.weight = weight
            self.bias = bias
            yield self
        finally:
            self.weight = old_weight
            self.bias = old_bias


class Dense(Linear, nn.Linear):
    pass

class Conv1d(Linear, nn.Conv1d):
    pass

class Conv2d(Linear, nn.Conv1d):
    pass

class Conv3d(Linear, nn.Conv1d):
    pass

