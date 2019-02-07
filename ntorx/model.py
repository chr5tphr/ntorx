#!/usr/bin/env python3
import torch
from torch.nn import Conv3d, MaxPool3d, ReLU, Linear, Sequential, Module
from collections import OrderedDict
from logging import getLogger

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, init
from torch.optim import Adam

from .nn import BatchView, Sequential

logger = getLogger(__name__)

class Parametric(Module):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dev = device

    def init_params(self):
        def _init(X):
            try:
                w = X.weight.data
            except AttributeError:
                pass
            else:
                init.normal_(w)

            try:
                b = X.bias.data
            except AttributeError:
                pass
            else:
                init.normal_(b)

        self.apply(_init)

    def device(self, dev=None):
        if dev is None:
            if self._dev is None:
                self._dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self._dev = dev
        return self._dev

    def save_params(self, fpath, *args, **kwargs):
        dest = fpath.format(*args, **kwargs)
        torch.save(self.state_dict(), dest)
        return dest

    def load_params(self, fpath, *args, **kwargs):
        src  = fpath.format(*args, **kwargs)
        self.load_state_dict(torch.load(src))

    def train_params(self):
        pass

    def test_params(self):
        pass

    def loss_params(self):
        pass

class SeqClassifier(Sequential, Parametric):
    def train_params(self, dataset, validset=None, bsize=1, nepochs=1, spath=None, start=0, sfreq=5, nslope=5, **kwargs):
        loader  = DataLoader(dataset, bsize, shuffle=True, num_workers=kwargs.get('num_workers', 0))
        nsampmx = len(dataset)

        self.to(self.device())

        lossfn = CrossEntropyLoss()
        lr = 1e-1
        optfn  = Adam
        opti = optfn(self.parameters(), lr=lr)

        self.train()
        valoss = []
        logger.info('Starting training...')
        for epoch in range(start, start+nepochs):
            # Optimize
            try:
                closs = 0.
                nsamp = 0
                for data, label in loader:
                    x = data.to(self.device())
                    t = label.to(self.device())

                    y = self(x)
                    loss = lossfn(y, t)
                    closs += loss.detach().item()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                    del loss

                    nsamp += len(data)
                    logger.info('Processed %d/%d samples...', nsamp, nsampmx)
                    del x, y, t

                logger.info('Epoch: %03d, train-loss: %.2e'%(epoch+1, closs/nsamp))

                if validset is not None:
                    # Compute validation loss and, if appropriate, update learning rate
                    cval = self.loss_params(validset, bsize)
                    valoss.append(cval)

                    logger.info('Epoch: %03d, valid-loss: %.2e'%(epoch, valoss[-1]))
                    if len(valoss) >= nslope:
                        a = np.arange(nslope)
                        a -= a.mean()
                        b = np.array(valoss[-nslope:])
                        b -= b.mean()
                        slope = (a*b).sum() / (a**2).sum()
                        logger.info('Epoch: %03d, slope: %.2e'%(epoch+1, slope))

                        if slope > -0.01:
                            valoss = []
                            lr *= .1
                            for gr in opti.param_groups:
                                gr['lr'] = lr
                else:
                    if not ((epoch + 1) % nslope):
                        lr *= .1
                        for gr in opti.param_groups:
                            gr['lr'] = lr
            except BaseException as err:
                dest = self.save_params(spath, epoch=epoch+1, error=type(err).__name__, **kwargs)
                logger.info('Emergency-saved parameters to \'{}\''.format(dest))
                raise err

            # Save model params
            if (not ((epoch + 1) % sfreq) or (epoch == start+nepochs-1)) and spath:
                dest = self.save_params(spath, epoch=epoch+1, **kwargs)
                logger.info('Saved parameters to \'{}\''.format(dest))
                if validset is not None:
                    acc = self.test_params(validset, bsize)
                    logger.info('Epoch: {:03d} , acc: {:.2e}'.format(epoch, acc))

    def test_params(self, dataset, bsize=1):
        loader = DataLoader(dataset, bsize, shuffle=False, num_workers=4)

        self.to(self.device())

        self.train(False)
        acc = 0
        nsamp = 0
        nsampmx = len(dataset)

        logger.info('Starting validation...')
        for data, label in loader:
            x = data.to(self.device())
            t = label.to(self.device())
            with torch.no_grad():
                y = self(x)

            acc += (y.detach().argmax(1) == t).sum()

            nsamp += len(data)
            logger.info('Processed %d/%d samples...', nsamp, nsampmx)
            del x, t, y
        acc /= len(dataset)
        return acc

    def loss_params(self, dataset, bsize=1, dev=None):
        loader = DataLoader(dataset, bsize, shuffle=False, num_workers=4)

        self.to(self.device())

        lossfn = CrossEntropyLoss()
        loss = 0

        self.train(False)
        acc = 0
        for data, label in loader:
            x = data.to(self.device())
            t = label.to(self.device())
            y = self(x)

            loss += lossfn(y, t).item()
        return loss/len(dataset)
