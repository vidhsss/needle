"""Optimization module"""
import needle as ndl
from collections import defaultdict
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
         for i, param in enumerate(self.params):
            if i not in self.u:
                self.u[i] = 0
            grad = ndl.Tensor(param.grad, dtype='float32').data + self.weight_decay * param.data       
            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * grad
            param.data = param.data - self.u[i] * self.lr
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


# class Adam(Optimizer):
#     def __init__(
#         self,
#         params,
#         lr=0.01,
#         beta1=0.9,
#         beta2=0.999,
#         eps=1e-8,
#         weight_decay=0.0,
#     ):
#         super().__init__(params)
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.weight_decay = weight_decay
#         self.t = 0

#         self.m = {}
#         self.v = {}

#     def step(self):
#         ### BEGIN YOUR SOLUTION
#         self.t += 1
#         for i, param in enumerate(self.params):
#             if i not in self.m:
#                 self.m[i] = ndl.init.zeros(1,dtype='float32')
#                 self.v[i] = ndl.init.zeros(1,dtype='float32')
#             grad = ndl.Tensor(param.grad, dtype='float32').data + param.data * self.weight_decay
#             # m_{t+1}, v{t+1}
#             self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
#             self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
#             # bias correction
#             m_hat = (self.m[i]) / (1 - self.beta1 ** self.t)
#             v_hat = (self.v[i]) / (1 - self.beta2 ** self.t)
#             m_hat= ndl.Tensor(m_hat,dtype="float32")
#             v_hat= ndl.Tensor(v_hat,dtype="float32")
#             param.data = param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps) 
#         ### END YOUR SOLUTION

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        
        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data
            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)
            unbiased_m = self.m[w] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[w] / (1 - self.beta2 ** self.t)
            w.data = w.data - self.lr * unbiased_m / (unbiased_v**0.5 + self.eps)
        # self.t+=1
        # for p in self.params:
        #   if id(p) not in self.m:
        #     self.m[id(p)] = ndl.init.zeros(1,dtype="float32")
        #     self.v[id(p)] = ndl.init.zeros(1,dtype="float32")
        #   grad = p.grad.data + self.weight_decay*p.data
        #   self.m[id(p)] = self.beta1* self.m[id(p)] + (1-self.beta1)*grad
        #   self.v[id(p)] = self.beta2*self.v[id(p)] + (1-self.beta2)*grad*grad
        #   bc_u = self.m[id(p)]/ (1-self.beta1**self.t)
        #   bc_v = self.v[id(p)]/ (1-self.beta2**self.t)
        #   bc_u = ndl.Tensor(bc_u,dtype="float32")
        #   bc_v = ndl.Tensor(bc_v,dtype="float32")
        #   p.data = p.data - self.lr * bc_u / (bc_v**0.5 + self.eps)