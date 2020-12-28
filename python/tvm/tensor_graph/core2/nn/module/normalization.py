from tvm.tensor_graph.core2.graph.abstract import TensorType
from tvm.tensor_graph.core2.graph.concrete import Tensor, FloatTensor, Compute
from tvm.tensor_graph.core2.nn import functional as F

from .base import StateTensor, Module


class BatchNorm2d(Module):
    """BatchNorm2d of float32 precision
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.alpha = FloatTensor(
            (num_features,), target="llvm", device=0, name="bn_alpha", requires_grad=True)
        self.beta = FloatTensor(
            (num_features,), target="llvm", device=0, name="bn_beta", requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.tracking_mean = StateTensor(TensorType(
                (num_features,)), target="llvm", device=0, name="bn_track_mean", requires_grad=False)
            self.tracking_var = StateTensor(TensorType(
                (num_features,)), target="llvm", device=0, name="bn_track_var", requires_grad=False)
        else:
            self.tracking_mean = None
            self.tracking_var = None

    def eval(self):
        self.track_running_stats = False
        self.tracking_mean = None
        self.tracking_var = None

    def train(self):
        self.track_running_stats = True
        self.tracking_mean = StateTensor(TensorType(
                (self.num_features,)), target="llvm", device=0, name="bn_track_mean", requires_grad=False)
        self.tracking_var = StateTensor(TensorType(
            (self.num_features,)), target="llvm", device=0, name="bn_track_var", requires_grad=False)

    def forward(self, inputs):
        requires_grad = inputs.requires_grad or self.requires_grad
        if self.track_running_stats:
            ret, mean, var = F.batch_norm2d(
                inputs, self.alpha, self.beta, self.eps, self.momentum,
                track_running_stats=self.track_running_stats, tracking_mean=self.tracking_mean,
                tracking_var=self.tracking_var, output_dtype="float32", requires_grad=requires_grad)
            self.tracking_mean.set_update(mean)
            self.tracking_var.set_update(var)
            return ret
        else:
            ret = F.batch_norm2d(
                inputs, self.alpha, self.beta, self.eps, self.momentum,
                track_running_stats=self.track_running_stats, tracking_mean=self.tracking_mean,
                tracking_var=self.tracking_var, output_dtype="float32", requires_grad=requires_grad)
            return ret
