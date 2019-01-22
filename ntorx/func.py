
class Mlist(list):
    '''
        Access elements of list as if accessing only a single element.
    '''

    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], list):
                return super().__new__(cls, args[0])
            else:
                return args[0]
        else:
            return super().__new__(cls, args)

    def __getattr__(self, name):
        ret = Mlist()
        for obj in self:
            ret.append(getattr(obj, name))
        return ret

    def __call__(self, *args, **kwargs):
        ret = Mlist()
        for obj in self:
            ret.append(obj(*args, **kwargs))
        return ret
