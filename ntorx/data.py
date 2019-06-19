#!/usr/bin/env python3
import numpy as np
import torch
import logging

from torch.utils.data import Dataset
from skimage.transform import resize

from .util import procvid

logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    def __init__(self, infodict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._info = infodict['data']
        self._ldict = infodict['label']

    def __len__(self):
        return len(self._info)

    def nclass(self):
        return len(self._ldict)

    def __getitem__(self, idx):
        raise NotImplementedError

class FFMPEGVideoDataset(VideoDataset):
    def __init__(self, *args, fps=24, frames=240, subframes=240, scale=(128, 128), crop=(112, 112), vtfn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fps = fps
        self._frames = frames
        self._scale = scale
        self._crop = crop
        self._vtfn = vtfn
        self._subframes = subframes

    def __getitem__(self, idx):
        obj = self._info[idx]

        video = procvid(obj['fpath'], nframes=self._frames, subframes=self._subframes, fps=self._fps, h=self._scale[0], w=self._scale[1], hc=self._crop[0], wc=self._crop[1])

        if self._vtfn is not None:
            video = self._vtfn(video)

        return video, self._ldict[obj['label']]['id']


class RandomCrop(object):
    def __init__(self, shape):
        self._shape = shape

    def __call__(self, data):
        H, W, C = data.shape
        h, w = self._shape
        t, l = [np.random.randint(0, n) for n in [H - h, W - w]]

        return data[t:t+h, l:l+w]

class RescaleShort(object):
    def __init__(self, ssize):
        self._ssize = ssize

    def __call__(self, data):
        H, W, C = data.shape
        s = self._ssize

        if H < W:
            h, w = s, W/H * s
        else:
            h, w = H/W * s, s

        return resize(data, (int(h), int(w)), anti_aliasing=True, mode='reflect')

class Flip(object):
    def __init__(self, axis):
        self._axis = axis

    def __call__(self, data):
        return np.flip(data, self._axis)

class ToTensor(object):
    def __call__(self, data):
        return torch.from_numpy(data.transpose((2, 0, 1)))

class ToVTensor(object):
    def __init__(self, device=None):
        self.device = device

    def __call__(self, data):
        tensor = torch.from_numpy(data.transpose((3, 0, 1, 2)).astype(np.float32))/255.
        tensor = tensor.to(torch.float32)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor
