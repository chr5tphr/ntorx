import numpy as np

from logging import getLogger
from math import ceil

logger = getLogger(__name__)


cmaps = {}
def register_cmap(func):
    cmaps[func.__name__] = func
    return func

@register_cmap
def gray(x):
    return np.stack([x]*3, axis=-1).clip(0., 1.)

@register_cmap
def wred(x):
    return np.stack([0.*x+1., 1.-x, 1.-x], axis=-1).clip(0., 1.)

@register_cmap
def wblue(x):
    return np.stack([1.-x, 1.-x, 0*x+1.], axis=-1).clip(0., 1.)

@register_cmap
def hot(x):
    return np.stack([x*3, x*3-1, x*3-2], axis=-1).clip(0., 1.)

@register_cmap
def cold(x):
    return np.stack([0.*x, x*2-1, x*2], axis=-1).clip(0., 1.)

@register_cmap
def coldnhot(x):
    hpos = hot((2*x-1.).clip(0., 1.))
    hneg = cold(-(2*x-1.).clip(-1., 0.))
    return hpos + hneg

@register_cmap
def bwr(x):
    hpos = wred((2*x-1.).clip(0., 1.))
    hneg = wblue(-(2*x-1.).clip(-1., 0.))
    return hpos + hneg - 1.

def colorize(im, bbox=None, center=None, cmap='hot'):
    if not isinstance(im, np.ndarray):
        raise TypeError("Only numpy arrays are supported!")

    if bbox is None:
        lo, hi = im.min(), im.max()
    else:
        lo, hi = bbox

    if center is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            lodat = (im.clip(lo, center) - lo) / (center - lo) - 1.
            lodat[~np.isfinite(lodat)] = 0.
            hidat = (im.clip(center, hi) - center) / (hi - center)
            hidat[~np.isfinite(hidat)] = 0.
        ndat = ((hidat + lodat) + 1.) / 2.
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            ndat = ((im - lo)/(hi-lo))
            ndat[~np.isfinite(ndat)] = 0.

    return cmaps[cmap](ndat.clip(0., 1.))

def montage(im, shape=None, filler=0.):
    if not isinstance(im, np.ndarray):
        raise TypeError("Only numpy arrays are supported!")
    if len(im.shape) not in (3, 4):
        raise TypeError('For a montage the array has to have either 3 (grayscale) or 4 (rgb) axes!')

    # add missing axis if omitted, squeeze it later
    squeeze = False
    if len(im.shape) == 3:
        im = im[(slice(None),)*3 + (None,)]
        squeeze = True

    N, H, W, C = im.shape
    if shape is None:
        w = h = ceil(N**.5)
    else:
        h, w = shape

    ret = np.full((h * w, H, W, C), filler)
    dim = min(N, h * w)
    ret[:dim] = im[:dim]
    ret = ret.reshape(h, w, H, W, C).transpose(0, 2, 1, 3, 4).reshape(h * H, w * W, C)

    if squeeze:
        ret = ret.squeeze(2)

    return ret

def imgify(im, bbox=None):
    if not isinstance(im, np.ndarray):
        raise TypeError('Only numpy arrays are supported!')
    if len(im.shape) not in (2, 3):
        raise TypeError('Input has to have either 2 or 3 axes!')
    if (len(im.shape) == 3) and (im.shape[2] not in (1, 3, 4)):
        raise TypeError('Last axis of input are color channels, which have to either be 1, 3, 4 or be omitted entirely!')

    # rescale data if necessary
    if im.dtype != np.uint8:
        if bbox is None:
            lo, hi = im.min(), im.max()
        else:
            lo, hi = bbox
        im = ((im - lo) * 255/(hi-lo)).clip(0, 255).astype(np.uint8)
    # add missing axis if omitted
    if len(im.shape) == 2:
        im = im[:,:,None]
    return im

