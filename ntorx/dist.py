#!/usr/bin/env python3
import torch
import numpy as np

from itertools import product
from torch.nn import Module, Conv3d

def convsplit(shape, indices, ksize, stride, pad):
    tlen = len(shape)
    assert all([len(elem) == tlen for elem in [indices, ksize, stride, pad]])

    for dshp, dind, dksz, dstr, dpad in zip(shape, indices, ksize, stride, pad):
        pass

def rng_overhang(rng, ksize):
    left  = ksize//2 - (rng.start % rng.step)
    right = ksize//2 - ((rng.stop - 1) % rng.step)
    loh = range(rng.start - left, rng.start)
    roh = range(rng.stop, rng.stop + right)
    return loh, roh

def _outerhanginds(start, stop, ksize, stride):
    left  = ksize//2 - (start % stride)
    right = ksize//2 - ((stop - 1) % stride)
    return left, right

def _innerhanginds(start, stop, ksize, stride):
    left  = (start % stride) - stride + ksize//2 + 1
    right = (stop % stride) - stride + ksize//2 + 1
    return left, right

def _outerhang(slic, ksize, length, stride=1):
    # the values outside own that have to be received
    start, stop, _ = slic.indices(length)
    left, right = _outerhanginds(start, stop, ksize, stride)
    loh = slice(max(0, start - left), start)
    roh = slice(stop, min(length, stop + right))
    return loh, roh

def _innerhang(slic, ksize, length, stride=1):
    # the values inside own that have to be sent
    start, stop, _ = slic.indices(length)
    left, right = _innerhanginds(start, stop, ksize, stride)
    lih = slice(start, max(0, start + left) if start > 0 else 0)
    rih = slice(min(length, stop - right) if stop < length else length, stop)
    return lih, rih

def _outerarea(slic, ksize, length, stride=1):
    start, stop, _ = slic.indices(length)
    left, right = _outerhanginds(start, stop, ksize, stride)
    return slice(max(0, start - left), min(length, stop + right))

def _indhangprod(slices, ksize, shape, strides, func):
    return list(product(*(
        (lambda lh, rh: (lh, slc, rh))(*func(slc, ksz, shp, std))
        for slc, ksz, shp, std in zip(slices, ksize, shape, strides)
    )))

def outerhang(slices, ksize, shape, strides):
    return _indhangprod(slices, ksize, shape, strides, _outerhang)

def innerhang(slices, ksize, shape, strides):
    return _indhangprod(slices, ksize, shape, strides, _innerhang)

def indsplit(shape, splits):
    return list(product(*(
        tuple(
            slice(beg, end, 1)
            for beg, end in zip(range(0, dim, dim//num), range(dim//num, dim+1, dim//num))
        )
        for n, (dim, num) in enumerate(zip(shape, splits))
    )))

def hangindsplit(shape, splits, ksize, strides):
    return list(product(*(
        tuple(
            _outerarea(slice(beg, end), ksz, shp, std)
            for beg, end, ksz, shp, std in zip(range(0, dim, dim//num), range(dim//num, dim+1, dim//num), ksize, shape, strides)
        )
        for n, (dim, num) in enumerate(zip(shape, splits))
    )))

class ConvSplitExchange(Module):
    def __init__(self, rank):
        pass
