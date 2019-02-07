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
    return torch.eye(K, dtype=dtype, device=dtype)[x.flat].
