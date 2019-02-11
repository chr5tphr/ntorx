import torch

from torch.nn import Module
from torch import autograd

from .nn import Linear, Sequential

class Attributor(Module):
    @classmethod
    def of(clss, ttype):
        return type('%s%s'%(clss.__name__, ttype.__name__), (clss, ttype), {})

    def attribution(self, out):
        raise NotImplementedError()

class SequentialAttributor(Sequential, Attributor):
    def attribution(self, out):
        for module in reversed(self._modules.values()):
            assert isinstance(module, Attributor)
            out = module.attribution(out)

class PassthroughAttributor(Attributor):
    def attribution(self, out):
        return out

class ShapeAttributor(Attributor):
    def forward(self, x):
        ishape = x.shape
        ret = super().forward(x)
        self._ishape = ishape
        return ret

    def attribution(self, out):
        return out.reshape(self._ishape)

class PiecewiseLinearAttributor(Linear, Attributor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in = None

    def forward(self, x):
        self._in = x
        return super().forward(x)

    @classmethod
    def of(clss, ttype):
        assert issubclass(ttype, Linear)
        return super().of(ttype)

class LRPAlphaBeta(PiecewiseLinearAttributor):
    def __init__(self, *args, alpha=1, beta=0, use_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._use_bias = use_bias

    def attribution(self, out):
        R = out
        a = self._in
        alpha = self._alpha
        beta = self._beta

        weight = self.weight.data
        wplus  = torch.clamp(weight, max=0.)
        wminus = torch.clamp(weight, min=0.)

        bplus = None
        bminus = None
        if self._use_bias is not None:
            bias   = self.bias.data
            bplus  = torch.clamp(bias, max=0.)
            bminus = torch.clamp(bias, min=0.)

        a.requires_grad_()

        with self.with_params(wplus, bplus) as swap:
            zplus = swap(a)
        cplus, = autograd.grad(zplus, a, grad_outputs=alpha*R/(zplus + (zplus == 0.)), retain_graph=True)

        with self.with_params(wminus, bminus) as swap:
            zminus = swap(a)
        cminus, = autograd.grad(zminus, a, grad_outputs=beta*R/(zminus + (zminus == 0.)), retain_graph=True)

        return a*(cplus - cminus)

class DTDZPlus(PiecewiseLinearAttributor):
    def __init__(self, *args, use_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_bias = use_bias

    def attribution(self, out):
        R = out
        a = self._in

        weight = self.weight.data
        wplus  = torch.clamp(weight, max=0.)

        bplus = None
        if self._use_bias is not None:
            bias = self.bias.data
            bplus  = torch.clamp(bias, max=0.)

        a.requires_grad_()

        with self.with_params(wplus, bplus) as swap:
            zplus = swap(a)
        cplus, = autograd.grad(zplus, a, grad_outputs=R/(zplus + (zplus == 0.)), retain_graph=True)

        return a*cplus

class DTDWSquare(PiecewiseLinearAttributor):
    def __init__(self, *args, use_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_bias = use_bias

    def attribution(self, out):
        R = out
        a = self._in

        weight = self.weight
        wsquare = weight**2

        bplus = None
        if self._use_bias is not None:
            bias = self.bias
            bsquare = bias**2

        a.requires_grad_()

        with self.with_params(wplus, bplus) as swap:
            z = swap(a)
        c, = autograd.grad(z, a, grad_outputs=R/(z + (z == 0.)), retain_graph=True)

        return c

class DTDZB(PiecewiseLinearAttributor):
    def __init__(self, *args, lo=0, hi=1, use_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._lo = lo
        self._hi = hi
        self._use_bias = use_bias

    def attribution(self, out, **kwargs):
        R = out
        a = self._in
        lo = self._lo
        hi = self._hi

        weight = self.weight.data
        wplus  = torch.clamp(weight, max=0.)
        wminus = torch.clamp(weight, min=0.)

        bias   = None
        bplus  = None
        bminus = None
        if self._use_bias is not None:
            bias   = self.bias.data
            bplus  = torch.clamp(bias, max=0.)
            bminus = torch.clamp(bias, min=0.)

        upper = torch.ones_like(a)*hi
        lower = torch.ones_like(a)*lo
        a.requires_grad_()
        upper.requires_grad_()
        lower.requires_grad_()

        z = linear(a)

        with self.with_params(wplus, bplus) as swap:
            zplus = swap(a)

        with self.with_params(wminus, bminus) as swap:
            zminus = swap(a)

        zlh = z - zplus - zminus
        zlh.backward(R/(zlh + (zlh == 0.)))
        return a*a.grad + upper*upper.grad + lower*lower.grad

