import torch

class Mlist(list):
    """Access elements of list as if accessing only a single element.
    """

    def __new__(cls, *args):
        """Will either return an Mlist, or the object itself if the list only has a single element
        """
        if len(args) == 1:
            if isinstance(args[0], list):
                return super().__new__(cls, args[0])
            else:
                return args[0]
        else:
            return super().__new__(cls, args)

    def __getattr__(self, name):
        """Attributes are an Mlist of all the attributes of the elements
        """
        ret = Mlist()
        for obj in self:
            ret.append(getattr(obj, name))
        return ret

    def __call__(self, *args, **kwargs):
        """Will call all the elements of the Mlist and create an Mlist of return values
        """
        ret = Mlist()
        for obj in self:
            ret.append(obj(*args, **kwargs))
        return ret

def one_hot(x, K, dtype=None, device=None):
    """Encodes a PyTorch Tensor to One-Hot Representation

    :param x: some Tensor
    :type x: :py:class:`numpy.ndarray`[int]
    :param int K: size of last dimension of one-hot encoded
    :return: x one-hot encoded in last dimension
    :rtype: :py:class:`torch.Tensor`
    """
    return torch.eye(K, dtype=dtype, device=dtype)[x.flat]

def zdiv(a, b, out=None):
    """Element-wise divides tensor a by non-zero elements of b. Division by zero result indicies of out are left untouched (zero by default)

    :param a: numerator
    :type a: :py:class:`torch.Tensor`
    :param b: denominator
    :type b: :py:class:`torch.Tensor`
    :param out: tensor where the result is stored
    :type out: :py:class:`torch.Tensor` or None
    :rtype b: :py:class:`torch.Tensor`
    """
    if out is None:
        out = torch.zeros_like(a)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out

def zdiv_(a, b):
    """Element-wise divides tensor a in-place by non-zero elements of b. Division by zero indicies of a are left untouched.

    :param a: numerator
    :type a: :py:class:`torch.Tensor`
    :param b: denominator
    :type b: :py:class:`torch.Tensor`
    :rtype b: :py:class:`torch.Tensor`
    """
    mask = b != 0
    a[mask] = a[mask] / b[mask]
    return a


def softplus_relu_diff(x, beta):
    return torch.log(1. + torch.exp(-beta*torch.abs(x))) / beta

class Softplus(torch.autograd.Function):
    """
        Note that for beta ~ 1e-5 the gradient is not stable
    """
    @staticmethod
    def forward(ctx, x, beta):
        bcast = beta.clamp(1e-5)[(None, slice(None)) + (None,)*(len(x.shape)-2)]
        # ln(1 + exp(-beta*|x|) ) / beta
        npwr = x.abs().neg_().mul_(bcast).exp_().log1p_().div_(bcast)
        out = torch.nn.functional.relu(x).add_(npwr)
        ctx.save_for_backward(x, beta)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, beta = ctx.saved_tensors
        bcast = beta.clamp(1e-5)[(None, slice(None)) + (None,)*(len(x.shape)-2)]
        xnabs = x.abs().neg_()
        npwr = xnabs.mul(bcast).exp_()
        npwr1p = 1. + npwr

        grad_x = x.sign().neg_().mul_(npwr).div_(npwr1p)
        grad_x += torch.nn.functional.relu(x).sign_()
        grad_x *= grad_out

        grad_beta = xnabs.mul_(npwr).div_(npwr1p)
        grad_beta += npwr.log1p_().div_(bcast).neg_()
        grad_beta /= bcast
        grad_beta *= grad_out
        grad_beta = grad_beta.sum((0,) + tuple(range(2, len(x.shape))) )
        return grad_x, grad_beta
