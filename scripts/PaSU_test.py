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
from torch.nn import Dropout, MaxPool2d, BatchNorm2d, AdaptiveAvgPool2d
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.models.vgg import cfg as vggconfig
from torchvision.transforms import Pad, ToTensor, Compose
from torch import nn

from ntorx.attribution import DTDZPlus, DTDZB, ShapeAttributor, SequentialAttributor, PassthroughAttributor, GradientAttributor
from ntorx.image import colorize, montage, imgify
from ntorx.model import Parametric, FeedForwardParametric
from ntorx.nn import Dense, BatchView, PaSU, Conv2d, Sequential
from ntorx.util import config_logger


logger = getLogger()

def xdg_data_home():
    return environ.get('XDG_DATA_HOME', path.join(environ['HOME'], '.local', 'share'))


class FeedFwd(Sequential, FeedForwardParametric):
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

class VGG16(FeedForwardParametric):
    def __init__(self, in_dim, out_dim, relu=True, beta=20, init_weights=True):
        super().__init__()
        def make_layers(cfg, batch_norm=False):
            layers = []
            in_channels = in_dim
            for v in cfg:
                if v == 'M':
                    layers += [MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, BatchNorm2d(v), PaSU(v, relu=relu, init=beta)]
                    else:
                        layers += [conv2d, PaSU(v, relu=relu, init=beta)]
                    in_channels = v
            return Sequential(*layers)
        self.features = make_layers(vggconfig['D'])
        self.avgpool = AdaptiveAvgPool2d((7, 7))
        self.classifier = Sequential(
            Dense(512 * 7 * 7, 4096),
            PaSU(4096, relu=relu, init=beta),
            Dropout(),
            Dense(4096, 4096),
            PaSU(4096, relu=relu, init=beta),
            Dropout(),
            Dense(4096, out_dim),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Dense):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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
    #dataset = MNIST(root=ctx.obj.data, train=True , transform=Compose([Pad(2), ToTensor()]), download=ctx.obj.download)
    dataset = CIFAR10(root=ctx.obj.data, train=True , transform=Compose([ToTensor()]), download=ctx.obj.download)
    loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=ctx.obj.workers)

    #model = FeedFwd((1, 32, 32), 10, relu=force_relu, beta=beta)
    model = VGG16(3, 10, relu=force_relu, beta=beta)
    if load is not None:
        model.load_params(load)
    model.device(ctx.obj.device)

    optargs = []
    wparams = chain(*[module.parameters() for module in model.modules() if not isinstance(module, (FeedForwardParametric, torch.nn.Sequential, PaSU))])
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
    #dataset = MNIST(root=ctx.obj.data, train=True , transform=Compose([Pad(2), ToTensor()]), download=ctx.obj.download)
    dataset = CIFAR10(root=ctx.obj.data, train=True , transform=Compose([ToTensor()]), download=ctx.obj.download)
    loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=ctx.obj.workers)

    #model = FeedFwd((1, 32, 32), 10, relu=force_relu, beta=beta)
    model = VGG16(3, 10, relu=force_relu, beta=beta)
    if load is not None:
        model.load_params(load)
    model.device(ctx.obj.device)

    optargs = []
    if not fix_weights:
        wparams = chain(*[module.parameters() for module in model.modules() if not isinstance(module, (FeedForwardParametric, torch.nn.Sequential, PaSU))])
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
    #dataset = MNIST(root=ctx.obj.data, train=True , transform=Compose([Pad(2), ToTensor()]), download=ctx.obj.download)
    dataset = CIFAR10(root=ctx.obj.data, train=True , transform=Compose([ToTensor()]), download=ctx.obj.download)
    loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=ctx.obj.workers)

    #model = FeedFwd((1, 32, 32), 10, relu=force_relu, beta=beta)
    model = VGG16(3, 10, relu=force_relu, beta=beta)
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
    #dataset = MNIST(root=ctx.obj.data, train=True , transform=Compose([Pad(2), ToTensor()]), download=ctx.obj.download)
    dataset = CIFAR10(root=ctx.obj.data, train=True , transform=Compose([ToTensor()]), download=ctx.obj.download)
    loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=ctx.obj.workers)

    #model = FeedFwd((1, 32, 32), 10, relu=force_relu, beta=beta)
    model = VGG16(3, 10, relu=force_relu, beta=beta)
    if load is not None:
        model.load_params(load)
    else:
        logger.warning('Using random model parameters for attribution!')
    model.to(model.device(ctx.obj.device))

    data, label = next(iter(loader))
    data = data.to(ctx.obj.device)
    #attrib = model.attribution(model(data))
    attrib = model.attribution(inpt=data)

    carr = np.moveaxis(attrib.detach().cpu().numpy(), 1, -1)
    carr /= np.abs(carr).sum((1, 2, 3), keepdims=True)
    if output.isatty():
        imprint(colorize(carr.squeeze(3), cmap='bwr'), montage=True)
    else:
        img = colorize(montage(carr).squeeze(2), cmap='wred')
        Image.fromarray(imgify(img)).save(output, format='png')

if __name__ == '__main__':
    main()
