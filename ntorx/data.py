#!/usr/bin/env python3
import numpy as np
import torch
import logging

from torch.utils.data import Dataset
from skimage.transform import resize
from imageio import get_reader

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
        obj = self._info[idx]
        with get_reader('%s'%obj['fpath'], format='avbin') as reader:
            meta  = reader.get_meta_data()
            video = np.empty([meta['nframes']] + list(meta['size'])[::-1] + [3], dtype=np.uint8)
            for frame in video:
                reader.get_next_data(out=frame)
        return video, self._ldict[obj['label']]['id']

class TransfVidDataset(VideoDataset):
    def __init__(self, *args, fps=24.0, imtransformer=lambda x,y: None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fps = fps
        self.imtransformer = imtransformer

    def __getitem__(self, idx):
        obj = self._info[idx]
        with get_reader('%s'%obj['fpath'], format='avbin') as reader:
            meta  = reader.get_meta_data()
            # target frame step
            tfstep = 1./self._fps
            # source frame step
            sfstep = 1./meta['fps']
            #target time
            ttime = 0.
            # video buffer
            vbuf = reader.create_empty_image()
            # source time, 0.5 times the source frame step such that when the real source and target time
            # overlap, float inaccuracies do not lead to one copied and one skipped frame in the target
            stime = 0.5 * sfstep
            # last target frame
            lframe = None

            # number of target frames
            ntframes = int(meta['nframes'] * sfstep / tfstep)
            # transform buffer such that we get the target shape
            tshape = self.imtransformer(vbuf).shape

            # target video
            video = np.empty([ntframes] + list(tshape), dtype=np.uint8)
            for frame in video:
                # read frames until we surpass the target time
                while stime <= ttime:
                    reader.get_next_data(out=vbuf)
                    lframe = None
                    stime += sfstep

                # transform source frame or use the last transformed frame if we did not read a new source frame during this iteration
                if lframe is None:
                    frame[:] = self.imtransformer(vbuf)
                    lframe = frame
                else:
                    frame[:] = lframe
                ttime += tfstep
        return video, self._ldict[obj['label']]

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
        #for tnum in range(5):
        #    try:
        #        video = procvid(obj['fpath'], nframes=self._frames, fps=self._fps, h=self._scale[0], w=self._scale[1], hc=self._crop[0], wc=self._crop[1])
        #    except ValueError:
        #        logger.warning('Could not read file, retrying... : %s'%obj['fpath'])
        #    else:
        #        break
        #    if tnum >= 4:
        #        raise RuntimeError('Unable to read file: \'%s\''%obj['fpath'])

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
