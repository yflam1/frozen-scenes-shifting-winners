class AnyBase(object):
    @property
    def cname_(self) -> str:
        return self.__class__.__name__
