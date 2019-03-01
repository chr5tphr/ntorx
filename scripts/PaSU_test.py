from argparse import Namespace
from collections import OrderedDict
from itertools import chain
from logging import getLogger
from os import path, environ
from sys import stdout

import click
import numpy as np
import torch

from PIL import Image
from tctim import imprint
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Pad, ToTensor, Compose
from torch import nn

from ntorx.attribution import DTDZPlus, DTDZB, ShapeAttributor, SequentialAttributor, PassthroughAttributor, GradientAttributor
from ntorx.image import colorize, montage, imgify
from ntorx.model import Parametric, SequentialParametric
from ntorx.nn import Dense, BatchView, PaSU
from ntorx.util import config_logger


logger = getLogger()

def xdg_data_home():
    return environ.get('XDG_DATA_HOME', path.join(environ['HOME'], '.local', 'share'))


class FeedFwd(SequentialAttributor, SequentialParametric):
    def __init__(self, in_dim, out_dim, relu=True, beta=1e2):
        in_flat = np.prod(in_dim)
        ABatchView = ShapeAttributor.of(BatchView)
        BDense = DTDZB.of(Dense)
        PDense = DTDZPlus.of(Dense)
        PPaSU  = PassthroughAttributor.of(PaSU)
        super().__init__(
            OrderedDict([
                ('view0', ABatchView(in_flat)),
                ('dens1', BDense(in_flat,    1024, lo=-1., hi=1.)),
                ('actv1', PPaSU(1024, relu=relu, init=beta)),
                ('dens2', PDense(   1024,    1024)),
                ('actv2', PPaSU(1024, relu=relu, init=beta)),
                ('dens3', PDense(   1024,    1024)),
                ('actv3', PPaSU(1024, relu=relu, init=beta)),
                ('dens4', PDense(   1024, out_dim)),
            ])
        )

class GradFeedFwd(GradientAttributor, SequentialParametric):
    def __init__(self, in_dim, out_dim, relu=True, beta=1e2):
        in_flat = np.prod(in_dim)
        super().__init__(
            OrderedDict([
                ('view0', BatchView(in_flat)),
                ('dens1', Dense(in_flat,    1024)),
                ('actv1', PaSU(1024, relu=relu, init=beta)),
                ('dens2', Dense(   1024,    1024)),
                ('actv2', PaSU(1024, relu=relu, init=beta)),
                ('dens3', Dense(   1024,    1024)),
                ('actv3', PaSU(1024, relu=relu, init=beta)),
                ('dens4', Dense(   1024, out_dim)),
            ])
        )
#class VGG16(SequentialAttributor, SequentialParametric):
#    def __init__(self, in_dim, out_dim, init_weights=True):
#        ABatchView = ShapeAttributor.of(BatchView)
#        BConv  = DTDZB.of(Conv2d)
#        PConv  = DTDZPlus.of(Conv2d)
#        PDense = DTDZPlus.of(Dense)
#        PPaSU  = PassthroughAttributor.of(PaSU)
#        PDrop  = PassthroughAttributor.of(nn.Dropout)
#
#        def _conv_unit(in_filters, out_filters, suffix='?', num_conv=3, BaseConv=PConv2d):
#            contup = lambda sfix, base, in_channels: OrderedDict([
#                ('conv%s'%sfix, base(in_channels, out_filters, kernel_size=3, padding=1)),
#                ('actv%s'%sfix, PPaSU(out_filters))
#            ])
#            retval = OrderedDict([
#                contup('conv%s.0'%suffix, BaseConv, in_filters),
#            ])
#
#            for it in range(conv_)
#            return OrderedDict([
#                ('conv%s.%2d'%(suffix, it), (BaseConv if it == 0 else PConv2d)) for it in range(num_conv)
#            ]) + OrderedDict([
#                ('pool%s'%suffix, MaxPool2d(kernel_size=2, stride=2)),
#            ])
#
#        features = SequentialAttributor(
#            _conv_unit()
#            OrderedDict([
#
#            ])
#        )
#        classifier = SequentialAttributor(
#            OrderedDict([
#                ('dens1', PDense(512 * 7 * 7, 4096)),
#                ('actv1', PPaSU(4096, relu=relu, init=beta)),
#                ('drop1', PDrop()),
#                ('dens1', PDense(4096, 4096)),
#                ('actv1', PPaSU(4096, relu=relu, init=beta)),
#                ('drop1', PDrop()),
#                ('dens1', PDense(4096, out_dim)),
#            ])
#        )
#
#    def _initialize_weights(self):
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                if m.bias is not None:
#                    nn.init.constant_(m.bias, 0)
#            elif isinstance(m, nn.Linear):
#                nn.init.normal_(m.weight, 0, 0.01)
#                nn.init.constant_(m.bias, 0)

@click.group()
@click.option('--log', type=click.File(), default=stdout)
@click.option('--threads', type=int, default=0)
@click.option('--workers', type=int, default=4)
@click.option('--download/--no-download', default=False)
@click.option('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
@click.option('--datapath', default=path.join(xdg_data_home(), 'dataset'))
@click.pass_context
def main(ctx, log, threads, workers, download, device, datapath):
    torch.set_num_threads(threads)
    config_logger(log)

    ctx.ensure_object(Namespace)
    ctx.obj.download = download
    ctx.obj.device = torch.device(device)
    ctx.obj.data = datapath
    ctx.obj.workers = workers

@main.command()
@click.option('-c', '--checkpoint', type=click.Path())
@click.option('-l', '--load', type=click.Path())
@click.option('-s', '--start', type=int, default=0)
@click.option('-n', '--nepochs', type=int, default=10)
@click.option('-b', '--bsize', type=int, default=32)
@click.option('-f', '--sfreq', type=int, default=1)
@click.option('--nslope', type=int, default=5)
@click.option('--lr', type=float, default=1e-3)
@click.option('--beta', type=float, default=1e2)
@click.option('--force-relu/--no-force-relu', default=False)
@click.pass_context
def train(ctx, checkpoint, load, start, nepochs, bsize, sfreq, nslope, lr, beta, force_relu):
    dataset = MNIST(root=ctx.obj.data, train=True , transform=Compose([Pad(2), ToTensor()]), download=ctx.obj.download)
    loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=ctx.obj.workers)

    model = FeedFwd((1, 32, 32), 10, relu=force_relu, beta=beta)
    if load is not None:
        model.load_params(load)
    model.device(ctx.obj.device)

    optargs = []
    wparams = chain(*[module.parameters() for module in model.modules() if not isinstance(module, (torch.nn.Sequential, PaSU))])
    optargs += [{'params': wparams, 'lr': lr}]
    optimizer = torch.optim.Adam(optargs)

    model.train_params(loader, optimizer, nepochs=nepochs, nslope=nslope, spath=checkpoint, start=start, sfreq=sfreq)

@main.command()
@click.option('-c', '--checkpoint', type=click.Path())
@click.option('-l', '--load', type=click.Path())
@click.option('-s', '--start', type=int, default=0)
@click.option('-n', '--nepochs', type=int, default=10)
@click.option('-b', '--bsize', type=int, default=32)
@click.option('-f', '--sfreq', type=int, default=1)
@click.option('--nslope', type=int, default=5)
@click.option('--lr', type=float, default=1e-3)
@click.option('--beta-decay', type=float, default=1e-3)
@click.option('--fix-weights/--no-fix-weights', default=True)
@click.pass_context
def betatune(ctx, checkpoint, load, start, nepochs, bsize, sfreq, nslope, lr, fix_weights, beta_decay):
    dataset = MNIST(root=ctx.obj.data, train=True , transform=Compose([Pad(2), ToTensor()]), download=ctx.obj.download)
    loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=ctx.obj.workers)

    model = FeedFwd((1, 32, 32), 10, relu=False)
    if load is not None:
        model.load_params(load)
    model.device(ctx.obj.device)

    optargs = []
    if not fix_weights:
        wparams = chain(*[module.parameters() for module in model.modules() if not isinstance(module, (torch.nn.Sequential, PaSU))])
        optargs += [{'params': wparams, 'lr': lr}]
    pparams = chain(*[module.parameters() for module in model.modules() if isinstance(module, PaSU)])
    optargs += [{'params': pparams, 'lr': lr, 'weight_decay': beta_decay}]
    optimizer = torch.optim.Adam(optargs)

    model.train_params(loader, optimizer, nepochs=nepochs, nslope=nslope, spath=checkpoint, start=start, sfreq=sfreq)

@main.command()
@click.option('-l', '--load', type=click.Path())
@click.option('-b', '--bsize', type=int, default=32)
@click.option('--force-relu/--no-force-relu', default=False)
@click.pass_context
def validate(ctx, load, bsize, force_relu):
    dataset = MNIST(root=ctx.obj.data, train=False, transform=Compose([Pad(2), ToTensor()]), download=ctx.obj.download)
    loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=4)

    model = FeedFwd((1, 32, 32), 10, relu=force_relu)
    if load is not None:
        model.load_params(load)
    else:
        logger.warning('Using random model parameters for validation!')
    model.device(ctx.obj.device)

    acc = model.test_params(loader)
    logger.info('Accuracy: {:.3f}'.format(acc))

@main.command()
@click.option('-l', '--load', type=click.Path())
@click.option('-b', '--bsize', type=int, default=32)
@click.option('-o', '--output', type=click.File(mode='wb'), default=stdout.buffer)
@click.option('--force-relu/--no-force-relu', default=False)
@click.pass_context
def attribution(ctx, load, bsize, output, force_relu):
    dataset = MNIST(root=ctx.obj.data, train=False, transform=Compose([Pad(2), ToTensor()]), download=ctx.obj.download)
    loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=4)

    model = GradFeedFwd((1, 32, 32), 10, relu=force_relu)
    if load is not None:
        model.load_params(load)
    else:
        logger.warning('Using random model parameters for attribution!')
    model.device(ctx.obj.device)

    data, label = next(iter(loader))
    attrib = model.attribution(model(data.to(ctx.obj.device)))

    carr = np.moveaxis(attrib.detach().numpy(), 1, -1)
    carr /= np.abs(carr).sum((1, 2, 3), keepdims=True)
    if output.isatty():
        imprint(colorize(carr.squeeze(3), cmap='bwr'), montage=True)
    else:
        img = colorize(montage(carr).squeeze(2), cmap='wred')
        Image.fromarray(imgify(img)).save(output, format='png')

if __name__ == '__main__':
    main()
