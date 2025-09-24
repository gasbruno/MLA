import math

class Module:
    """ Modules abstract class """

    @property
    def input( self ) -> list['Module']:
        return self.__input
    
    def __init__(self):
        print (self)

    def __str__(self):
        return "Operator -> " + self.__class__.__name__
    

class PlaceHolder(Module):
    """ PlaceHolder operator class """

    def __init__(self, value: float = 0):
        # The PlaceHolder operator has no inputs
        super().__init__()
        self.__value = value
    
    def forward(self):
        return self.__value

    def backward(self): pass

    def set(self, value: float):
        self.__value = value

    def __call__(self, *args, **kwds):
        return self.forward()
    
class Parameter(Module):
    """ Parameter operator class """

    def __init__(self, value: float = 0):
        # The Parameter operator has no inputs
        self.__value: float = value

    def forward(self):
        return self.__value

    def backward(self): pass

    def update(self, value: float):
        self.__value = value

    def __call__(self, *args, **kwds):
        return self.forward()

class Mul(Module):
    """ Multiply operator class """

    def __init__(self):
        super().__init__()
        self.__e1 = 0
        self.__e2 = 0
        self.__mul = 0
    
    def forward(self, e1: float, e2: float) -> float:
        self.__e1 = e1
        self.__e2 = e2
        self.__mul = e1 * e2
        return self.__mul

    def grad1(self, g: float) -> float:
        return g * self.__e2

    def grad2(self, g: float) -> float:
        return g * self.__e1
    
    def __call__(self, *args, **kwds):
        return self.forward(args[0], args[1])

class Add(Module):
    """ Add operator class """

    def __init__(self):
        super().__init__()
        self.__e1 = 0
        self.__e2 = 0
        self.__add = 0
    
    def forward(self, e1: float, e2: float) -> float:
        self.__e1 = e1
        self.__e2 = e2
        self.__add = e1 + e2
        return self.__add

    def grad1(self, g: float) -> float:
        return g

    def grad2(self, g: float) -> float:
        return g
    
    def __call__(self, *args, **kwds):
        return self.forward(args[0], args[1])
    

class Sub(Module):
    """ Sub operator class """

    def __init__(self):
        super().__init__()
        self.__e1 = 0
        self.__e2 = 0
        self.__sub = 0
    
    def forward(self, e1: float, e2: float) -> float:
        self.__e1 = e1
        self.__e2 = e2
        self.__sub = e1 - e2
        return self.__sub

    def grad1(self, g: float) -> float:
        return g

    def grad2(self, g: float) -> float:
        return -g
    
    def __call__(self, *args, **kwds):
        return self.forward(args[0], args[1])

class Square(Module):
    """ Square operator class """

    def __init__(self):
        super().__init__()

    def forward(self, e: float) -> float:
        self.__e = e
        self.__square = e ** 2
        return self.__square

    def backward(self, g: float) -> float:
        return 2 * g * self.__e
    
    def grad(self, g: float) -> float:
        return self.backward(g)

    def __call__(self, *args, **kwds):
        return self.forward(args[0])
    

class Relu(Module):
    def __init__(self):
        super().__init__()

    def forward(self, e: float) -> float:
        self.__e = e
        self.__relu = max(0, e)
        return self.__relu

    def backward(self, g: float) -> float:
        return g * (self.__e > 0)
    
    def grad(self, g: float) -> float:
        return self.backward(g)

    def __call__(self, *args, **kwds):
        return self.forward(args[0])


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def __sigmoid(self, v):
        return 1 / (1 + math.exp(-v))

    def forward(self, e: float) -> float:
        self.__e = e
        self.__sigmo = self.__sigmoid(e)
        return self.__sigmo

    def backward(self, g: float) -> float:
        return g * (self.__sigmo * (1 - self.__sigmo))
    
    def grad(self, g: float) -> float:
        return self.backward(g)

    def __call__(self, *args, **kwds):
        return self.forward(args[0])


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, e: float) -> float:
        self.__e = e
        self.__tan = math.tanh(e)
        return self.__tan

    def backward(self, g: float) -> float:
        return g * (1 - self.__tan ** 2)
    
    def grad(self, g: float) -> float:
        return self.backward(g)

    def __call__(self, *args, **kwds):
        return self.forward(args[0])
    
