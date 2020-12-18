import tvm
import numpy as np


# wrap tvm ndarray

class Array(object):
    def __init__(self, numpy_value, target, device):
        self.target = target
        self.device = device
        self.ctx = tvm.context(target, device)
        self.value = tvm.nd.array(numpy_value, self.ctx)
        self.shape = numpy_value.shape
        self.dtype = str(numpy_value.dtype)
    
    def astype(self, dtype):
        self.value = tvm.nd.array(self.value.asnumpy().astype(dtype), self.ctx)
        self.dtype = dtype