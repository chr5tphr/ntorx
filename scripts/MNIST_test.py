import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
from logging import getLogger

from torchvision import datasets, transforms

#from termex.layers.dense import Dense
#from termex.enums import ExplainingMethod, LRPRule
#from termex.networks import ExplainableSequential

from ntorx.nn import Dense, BatchView, PaSU
from ntorx.attribution import DTDZPlus, DTDZB, ShapeAttributor, SequentialAttributor, PassthroughAttributor
from ntorx.model import Parametric
from ntorx.util import config_logger

from tctim import imprint

logger = getLogger()

def mass_center(x):
    # X is (N, H, W), no channel!
    grid = torch.from_numpy(np.mgrid[:x.shape[1], :x.shape[2]][:, None]).to(device=x.device, dtype=x.dtype)
    return (x[None] * grid).sum(dim=(2,3)) / np.prod(x.shape[1:])

class FeedFwd(SequentialAttributor, Parametric):
    def __init__(self, in_dim, out_dim):
        in_flat = np.prod(in_dim)
        ABatchView = ShapeAttributor.of(BatchView)
        BDense = DTDZB.of(Dense)
        PDense = DTDZPlus.of(Dense)
        PPaSU  = PassthroughAttributor.of(PaSU)
        super().__init__(
            OrderedDict([
                ('view0', ABatchView(in_flat)),
                ('linr1', BDense(in_flat,    1024, lo=-1., hi=1.)),
                ('actv1', PPaSU(1024, relu=False)),
                ('linr2', PDense(   1024,    1024)),
                ('actv2', PPaSU(1024, relu=False)),
                ('linr3', PDense(   1024,    1024)),
                ('actv3', PPaSU(1024, relu=False)),
                ('linr4', PDense(   1024, out_dim)),
            ])
        )

def train(args, model, device, train_loader, optimizer, epoch, gamma=.5):
    model.train()
    final = None
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        #final = relv[0]
    #imprint(final.detach().cpu().numpy())

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def attribution(args, model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            relv = model.attribution(output.clamp(min=0.)).sum(dim=1)

            break
    imprint((lambda x: x / np.abs(x).sum(0))(relv[:2*7].cpu().numpy()).reshape(2,7,28,28).transpose(0,2,1,3).reshape(2*28,7*28))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='gamma (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    config_logger(sys.stdout)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = FeedFwd((1, 28, 28), 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer =

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, gamma=args.gamma)
        test(args, model, device, test_loader)
        attribution(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_ff.pt")

if __name__ == '__main__':
    main()
