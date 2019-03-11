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
from torch.nn import Dropout, MaxPool2d, AdaptiveAvgPool2d
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.models.vgg import cfg as vggconfig
from torchvision.transforms import Pad, ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomCrop
from torch import nn

from ntorx.attribution import DTDZPlus, DTDZB, ShapeAttributor, SequentialAttributor, PassthroughAttributor, GradientAttributor
from ntorx.image import colorize, montage, imgify
from ntorx.model import Parametric, FeedForwardParametric
from ntorx.nn import BatchView, PaSU, Sequential
from ntorx.util import config_logger

try:
    from ntorx.nn import Dense, Conv2d, BatchNorm2d
except ImportError:
    from ntorx.nn import __getattr__ as _getlin
    Dense = _getlin('Dense')
    Conv2d = _getlin('Conv2d')
    BatchNorm2d = _getlin('BatchNorm2d')

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
    def __init__(self, in_dim, out_dim, relu=True, beta=20, init_weights=True, batch_norm=False):
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
        self.features = make_layers(vggconfig['D'], batch_norm=batch_norm)
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

class ChoiceList(click.Choice):
    def __init__(self, *args, separator=',', **kwargs):
        super().__init__(*args, **kwargs)
        self.separator = separator

    def convert(self, values, param, ctx):
        retval = []
        for val in values.split(self.separator):
            retval.append(super(click.Choice, self).convert(val, param, ctx))
        return retval

@click.group(chain=True)
@click.option('--log', type=click.File(), default=stdout)
@click.option('-v', '--verbose', count=True)
@click.option('--threads', type=int, default=0)
@click.option('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
@click.pass_context
def main(ctx, log, verbose, threads, device):
    torch.set_num_threads(threads)
    config_logger(log, level='DEBUG' if verbose > 0 else 'INFO')

    ctx.ensure_object(Namespace)
    ctx.obj.device = torch.device(device)

@main.command()
@click.option('-l', '--load', type=click.Path())
@click.option('--beta', type=float, default=1e2)
@click.option('--force-relu/--no-force-relu', default=False)
@click.option('--batchnorm/--no-batchnorm', default=False)
@click.pass_context
def model(ctx, load, beta, force_relu, batchnorm):
    #model = FeedFwd((1, 32, 32), 10, relu=force_relu, beta=beta)
    model = GradientAttributor.of(VGG16)(3, 10, relu=force_relu, beta=beta, init_weights=load is None, batch_norm=batchnorm)
    if load is not None:
        model.load_params(load)
    model.device(ctx.obj.device)
    ctx.obj.model = model

@main.command()
@click.option('-b', '--bsize', type=int, default=32)
@click.option('--train/--test', default=True)
@click.option('--datapath', default=path.join(xdg_data_home(), 'dataset'))
@click.option('--dataset', type=click.Choice(['CIFAR10', 'MNIST']), default='CIFAR10')
@click.option('--download/--no-download', default=False)
@click.option('--workers', type=int, default=4)
@click.pass_context
def data(ctx, bsize, train, datapath, dataset, download, workers):
    if dataset == 'CIFAR10':
        transf = Compose(([RandomCrop(32, padding=4), RandomHorizontalFlip()] if train else []) + [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dset = CIFAR10(root=datapath, train=train , transform=transf, download=download)
    elif dataset == 'MNIST':
        dset = MNIST(root=data, train=train , transform=Compose([Pad(2), ToTensor()]), download=download)
    else:
        raise RuntimeError('No such dataset!')
    loader  = DataLoader(dset, bsize, shuffle=True, num_workers=workers)
    ctx.obj.loader = loader

@main.command()
@click.option('--param', type=ChoiceList(['weight', 'bias', 'beta', 'all']), default=['weight', 'bias'])
@click.option('--lr', type=float, default=1e-3)
@click.option('--decay', type=float)
@click.pass_context
def optimize(ctx, param, lr, decay):
    model = ctx.obj.model

    optargs = []
    if 'weight' in param:
        params = (module.weight for module in model.modules() if isinstance(module, Linear))
        arg = {'params': params, 'lr': lr}
        if decay:
            arg['weight_decay'] = decay
        optargs.append(arg)
    if 'bias' in param:
        params = (module.bias for module in model.modules() if isinstance(module, Linear))
        arg = {'params': params, 'lr': lr}
        if decay:
            arg['weight_decay'] = decay
        optargs.append(arg)
    if 'beta' in param:
        params = (module.beta for module in model.modules() if isinstance(module, PaSU))
        arg = {'params': params, 'lr': lr}
        if decay:
            arg['weight_decay'] = decay
        optargs.append(arg)
    if 'all' in param:
        arg = {'params': model.parameters(), 'lr': lr}
        if decay:
            arg['weight_decay'] = decay
        optargs.append(arg)

    ctx.obj.optimizer = torch.optim.Adam(optargs)

@main.command()
@click.option('-c', '--checkpoint', type=click.Path())
@click.option('-s', '--start', type=int, default=0)
@click.option('-n', '--nepochs', type=int, default=10)
@click.option('-f', '--sfreq', type=int, default=1)
@click.option('--nslope', type=int, default=5)
@click.pass_context
def train(ctx, checkpoint, start, nepochs, sfreq, nslope):
    loader = ctx.obj.loader
    model = ctx.obj.model
    optimizer = ctx.obj.optimizer

    model.train_params(loader, optimizer, nepochs=nepochs, nslope=nslope, spath=checkpoint, start=start, sfreq=sfreq)

@main.command()
@click.option('-o', '--output', type=click.File(mode='w'), default=stdout)
@click.pass_context
def validate(ctx, output):
    loader = ctx.obj.loader
    model = ctx.obj.model

    acc = model.test_params(loader)
    output.write('Accuracy: {:.3f}\n'.format(acc))

@main.command()
@click.option('-o', '--output', type=click.File(mode='w'), default=stdout)
@click.pass_context
def betastat(ctx, output):
    model = ctx.obj.model

    params = (module.beta for module in model.modules() if isinstance(module, PaSU))
    stats = [(param.data.mean().item(), param.data.std().item()) for param in params]

    head = '{: ^3} {: ^10} {: ^10}\n'.format('#', 'mean', 'std')
    info = '\n'.join('{: >3d} {: >10.2e} {: >10.2e}'.format(n, *stat) for n, stat in enumerate(stats)) + '\n'
    output.write(head + info)

@main.command()
@click.option('-o', '--output', type=click.File(mode='wb'), default=stdout.buffer)
@click.pass_context
def attribution(ctx, output):
    loader = ctx.obj.loader
    model = ctx.obj.model

    data, label = next(iter(loader))
    data = data.to(ctx.obj.device)
    #attrib = model.attribution(model(data))
    attrib = model.attribution(inpt=data)

    #carr = np.moveaxis(attrib.detach().cpu().numpy(), 1, -1)
    carr = np.abs(np.moveaxis(attrib.detach().cpu().numpy(), 1, -1)).sum(-1, keepdims=True)
    carr /= np.abs(carr).sum((1, 2, 3), keepdims=True)
    if output.isatty():
        imprint(colorize(carr.squeeze(3), cmap='bwr'), montage=True)
    else:
        img = colorize(montage(carr).squeeze(2), cmap='wred')
        Image.fromarray(imgify(img)).save(output, format='png')

if __name__ == '__main__':
    main(auto_envvar_prefix='PASU')
