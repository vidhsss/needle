"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, dtype=dtype,device=device))
        if bias:
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, dtype=dtype,device=device).reshape((1, out_features)))
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_out = X @ self.weight
        if self.bias:
          return X_out + self.bias.broadcast_to(X_out.shape)
        return X_out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        prod=1
        for i in X.shape:
            prod*=i
        return X.reshape((X.shape[0], prod // X.shape[0]))
        ### END YOUR SOLUTION

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y, device=logits.device)).sum()
        return (exp_sum - z_y_sum) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True,device=device,dtype=dtype))
        self.bias = Parameter(init.zeros(dim, requires_grad=True,device=device,dtype=dtype))
        self.running_mean = init.zeros(dim, requires_grad=False,device=device,dtype=dtype)
        self.running_var = init.ones(dim,requires_grad=False,device=device,dtype=dtype)
        ### END YOUR SOLUTION

# github 
    # def forward(self, x: Tensor) -> Tensor:
    #     ### BEGIN YOUR SOLUTION
    #     batch_size = x.shape[0]
    #     feature_size = x.shape[1]
    #     # running estimates
    #     mean = x.sum(axes=(0,)) / batch_size
    #     x_minus_mean = x - mean.broadcast_to(x.shape)
    #     var = (x_minus_mean ** 2).sum(axes=(0, )) / batch_size

    #     if self.training:
    #         self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
    #         self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

    #         x_std = ((var + self.eps) ** 0.5).broadcast_to(x.shape)
    #         normed = x_minus_mean / x_std
    #         return normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
    #     else:
    #         normed = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
    #         return normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
    #     ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          summed_x = ops.summation(x,(0,))
          mean_x = (summed_x/x.shape[0])
          sq = (x - mean_x.reshape((1,x.shape[1])).broadcast_to(x.shape)) ** 2
          var_x = (ops.summation(sq,(0,))/x.shape[0])
          std = (var_x.reshape((1,x.shape[1])).broadcast_to(x.shape) + self.eps) ** 0.5
          norm = (x - mean_x.reshape((1,x.shape[1])).broadcast_to(x.shape)) / std

          self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean_x.data
          self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var_x.data

          return ops.broadcast_to(self.weight,x.shape)*norm + ops.broadcast_to(self.bias,x.shape)
        else:
          norm = (x - self.running_mean.reshape((1,x.shape[1])).broadcast_to(x.shape)) / (self.running_var.reshape((1,x.shape[1])).broadcast_to(x.shape) + self.eps)**0.5
          return self.weight.reshape((1,x.shape[1])).broadcast_to(x.shape) * norm + self.bias.reshape((1,x.shape[1])).broadcast_to(x.shape)
      
class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim),requires_grad=True)
        self.bias = Parameter(init.zeros(dim),requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        mean = x.sum(axes=(1, )).reshape((batch_size, 1)) / feature_size
        x_minus_mean = x - mean.broadcast_to(x.shape)
        x_std = ((x_minus_mean ** 2).sum(axes=(1, )).reshape((batch_size, 1)) / feature_size + self.eps) ** 0.5
        norm = x_minus_mean / x_std.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x * (init.randb(*x.shape, p=(1 - self.p),dtype="float32")) / (1- self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION