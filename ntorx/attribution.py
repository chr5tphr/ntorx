import torch

from torch.nn import Module
from torch import autograd

from .nn import Linear, Sequential
from .func import zdiv

class Attributor(Module):
    @classmethod
    def of(cls, ttype):
        return type('%s%s'%(cls.__name__, ttype.__name__), (cls, ttype), {})

    @classmethod
    def cast(cls, obj, *args, **kwargs):
        assert isinstance(obj, Module)
        newcls = cls.of(obj.__class__)
        newobj = object.__new__(newcls)
        cls.__init__(newobj, *args, **kwargs)
        newobj.__dict__.update(obj.__dict__)
        return newobj

    def attribution(self, out):
        raise NotImplementedError()

class SequentialAttributor(Sequential, Attributor):
    def attribution(self, out):
        for module in reversed(self._modules.values()):
            assert isinstance(module, Attributor)
            out = module.attribution(out)
        return out

class GradientAttributor(Attributor):
    def forward(self, x):
        self._in = x
        return super().forward(x)

    def attribution(self, out=None, inpt=None):
        a = self._in if inpt is None else inpt
        a.requires_grad_()
        z = self(a).sum()
        out, = torch.autograd.grad(z, a, grad_outputs=out, retain_graph=True)
        return out

class SmoothGradAttributor(Attributor):
    def __init__(self, *args, std=1.0, niter=50, **kwargs):
        super().__init__(*args, **kwargs)
        self._std = std
        self._niter = niter

    def forward(self, x):
        self._in = x
        return super().forward(x)

    def attribution(self, out=None, inpt=None):
        a = self._in if inpt is None else inpt
        accu = torch.zeros_like(a)
        for n in range(self._niter):
            b = a + torch.normal(torch.zeros_like(a), torch.full_like(a, self._std))
            b.requires_grad_()
            z = self(b).sum()
            tout, = torch.autograd.grad(z, b, grad_outputs=out, retain_graph=True)
            accu += tout / self._niter
        return accu

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
    def of(cls, ttype):
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
        wplus  = torch.clamp(weight, min=0.)
        wminus = torch.clamp(weight, max=0.)

        bias   = self.bias.data
        if self._use_bias:
            bplus  = torch.clamp(bias, min=0.)
            bminus = torch.clamp(bias, max=0.)
        else:
            bplus = torch.zeros_like(bias)
            bminus = bplus

        a.requires_grad_()

        with self.with_params(wplus, bplus) as swap, autograd.enable_grad():
            zplus = swap(a)
        cplus, = autograd.grad(zplus, a, grad_outputs=alpha*zdiv(R, zplus), retain_graph=True)

        with self.with_params(wminus, bminus) as swap, autograd.enable_grad():
            zminus = swap(a)
        cminus, = autograd.grad(zminus, a, grad_outputs=beta*zdiv(R, zminus), retain_graph=True)

        return a*(cplus - cminus)

class LRPFlat(PiecewiseLinearAttributor):
    def attribution(self, out):
        R = out
        a = self._in

        weight = torch.ones_like(self.weight.data)
        bias = torch.zeros_like(self.bias.data)

        a = torch.ones_like(a, requires_grad=True)

        with self.with_params(weight, bias) as swap, autograd.enable_grad():
            z = swap(a)
        c, = autograd.grad(z, a, grad_outputs=zdiv(R, z), retain_graph=True)

        return c

class LRPEpsilon(PiecewiseLinearAttributor):
    def __init__(self, *args, epsilon=1e-1, use_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon
        self._use_bias = use_bias

    def attribution(self, out):
        R = out
        a = self._in
        epsilon = self._epsilon

        weight = self.weight.data

        bias   = self.bias.data
        if self._use_bias:
            bias = self.bias.data
        else:
            bias = torch.zeros_like(bias)

        a.requires_grad_()

        with self.with_params(weight, bias) as swap, autograd.enable_grad():
            z = swap(a)
        c, = autograd.grad(z, a, grad_outputs=zdiv(R, z + z.sign() * epsilon), retain_graph=True)

        return a*c

class DTDZPlus(PiecewiseLinearAttributor):
    def __init__(self, *args, use_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_bias = use_bias

    def attribution(self, out):
        R = out
        a = self._in

        weight = self.weight.data
        wplus  = torch.clamp(weight, min=0.)

        bias  = self.bias.data
        if self._use_bias:
            bplus = torch.clamp(bias, min=0.)
        else:
            bplus = torch.zeros_like(bias)

        a.requires_grad_()

        with self.with_params(wplus, bplus) as swap, autograd.enable_grad():
            zplus = swap(a)
        cplus, = autograd.grad(zplus, a, grad_outputs=zdiv(R, zplus), retain_graph=True)

        return a*cplus

class DTDWSquare(PiecewiseLinearAttributor):
    def __init__(self, *args, use_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_bias = use_bias

    def attribution(self, out):
        R = out
        a = self._in

        weight = self.weight.data
        wsquare = weight**2

        bias   = self.bias.data
        if self._use_bias:
            bsquare = bias**2
        else:
            bsquare = torch.zeros_like(bias)

        a = torch.ones_like(a, requires_grad=True)

        with self.with_params(wsquare, bsquare) as swap, autograd.enable_grad():
            z = swap(a)
        c, = autograd.grad(z, a, grad_outputs=zdiv(R, z), retain_graph=True)

        return c

class DTDZB(PiecewiseLinearAttributor):
    def __init__(self, *args, lo=-1., hi=1., use_bias=False, **kwargs):
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
        wplus  = torch.clamp(weight, min=0.)
        wminus = torch.clamp(weight, max=0.)

        bias   = self.bias.data
        if self._use_bias:
            bplus  = torch.clamp(bias, min=0.)
            bminus = torch.clamp(bias, max=0.)
        else:
            bias = torch.zeros_like(bias)
            bplus = bias
            bminus = bias

        upper = torch.full_like(a, hi, requires_grad=True)
        lower = torch.full_like(a, lo, requires_grad=True)
        a.requires_grad_()

        with autograd.enable_grad():
            z = self(a)

            with self.with_params(wplus, bplus) as swap:
                zplus = swap(lower)

            with self.with_params(wminus, bminus) as swap:
                zminus = swap(upper)

            zlh = z - zplus - zminus
        agrad, lgrad, ugrad = autograd.grad((zlh,), (a, lower, upper), grad_outputs=(zdiv(R, zlh),)*3, retain_graph=True)
        return a*agrad + lower*lgrad + upper*ugrad

class PoolingAttributor(Attributor):
    def __init__(self, *args, pool_op=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool_op = self.forward if pool_op is None else pool_op

    def forward(self, x):
        self._in = x
        return super().forward(x)

    def attribution(self, out):
        R = out
        a = self._in

        a.requires_grad_()
        with autograd.enable_grad():
            z = self._pool_op(a)
        cplus, = autograd.grad(z, a, grad_outputs=zdiv(R, z), retain_graph=True)

        return a*cplus

