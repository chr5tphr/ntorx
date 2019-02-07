from torch.nn import Module
from torch import autograd

from .nn import Linear, Sequential

class Attributor(Module):
    def attribution(self, *args, **kwargs):
        pass

class SequentialAttributor(Attributor):
    def __init__(self, sequential, *args, **kwargs):
        assert isinstance(sequential, Sequential)
        super().__init__(self, *args, **kwargs)

class PiecewiseLinearAttributor(Attibutor):
    def __init__(self, linear, activation, *args, **kwargs):
        assert isinstance(linear, Linear)
        super().__init__(self, *args, **kwargs)
        self._linear = linear
        self._activation = activation
        self._in = None

    def forward(self, x):
        self._in = x
        a = self.linear(x)
        z = self.activation(a)
        return z

class LRPAlphaBeta(PiecewiseLinearAttributor):
    def __init__(self, alpha=1, beta=0, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._alpha = alpha
        self._beta = beta

    def attribution(self, out, use_bias=False):
        R = out
        a = self._in
        linear = self._linear
        alpha = self._alpha
        beta = self._beta

        weight = linear.weight
        wplus = torch.maximum(0., weight)
        wminus = torch.minimum(0., weight)

        bplus = None
        bminus = None
        if use_bias is not None:
            bias = linear.bias
            bplus = torch.maximum(0., bias)
            bminus = torch.minimum(0., bias)

        a.requires_grad_()

        with linear.with_params(wplus, bplus) as swap:
            zplus = swap(a)
        cplus, = autograd.grad(zplus, a, grad_outputs=alpha*R/(zplus + (zplus == 0.)))

        with linear.with_params(wminus, bminus) as swap:
            zminus = swap(a)
        cminus, = autograd.grad(zminus, a, grad_outputs=beta*R/(zminus + (zminus == 0.)))

        return a*(cplus - cminus)

class DTDZPlus(PiecewiseLinearAttributor):
    def attribution(self, out, use_bias=False):
        R = out
        a = self._in
        linear = self.linear

        weight = linear.weight
        wplus = torch.maximum(0., weight)

        bplus = None
        if use_bias is not None:
            bias = linear.bias
            bplus = torch.maximum(0., bias)

        a.requires_grad_()

        with linear.with_params(wplus, bplus) as swap:
            zplus = swap(a)
        cplus, = autograd.grad(zplus, a, grad_outputs=R/(zplus + (zplus == 0.)))

        return a*cplus

class DTDWSquare(PiecewiseLinearAttributor):
    def attribution(self, out, use_bias=False):
        R = out
        a = self._in
        linear = self.linear

        weight = linear.weight
        wsquare = weight**2

        bplus = None
        if use_bias is not None:
            bias = linear.bias
            bsquare = torch.maximum(0., bias)

        a.requires_grad_()

        with linear.with_params(wplus, bplus) as swap:
            z = swap(a)
        c, = autograd.grad(z, a, grad_outputs=R/(z + (z == 0.)))

        return c

class DTDZB(PiecewiseLinearAttributor):
    def __init__(self, lo=1, hi=0, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._lo = lo
        self._hi = hi

    def attribution(self, out, lo=-1, hi=1, use_bias=False, **kwargs):
        if self._in is None:
            raise RuntimeError('Block has not yet executed forward_logged!')
        R = out
        a = self._in
        lo = self._lo
        hi = self._hi
        linear = self.linear

        weight = linear.weight
        wplus = torch.maximum(0., weight)
        wminus = torch.minimum(0., weight)

        bias = None
        bplus = None
        bminus = None
        if use_bias is not None:
            bias = linear.bias
            bplus = torch.maximum(0., bias)
            bminus = torch.minimum(0., bias)

        upper = torch.ones_like(a)*hi
        lower = torch.ones_like(a)*lo
        a.requires_grad_()
        upper.requires_grad_()
        lower.requires_grad_()

        z = linear(a)

        with linear.with_params(wplus, bplus) as swap:
            zplus = swap(a)

        with linear.with_params(wminus, bminus) as swap:
            zminus = swap(a)

        zlh = z - zplus - zminus
        zlh.backward(R/(zlh + (zlh == 0.)))
        return a*a.grad + upper*upper.grad + lower*lower.grad

