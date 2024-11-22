from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    
    def compute(self, Z):
        ## BEGIN YOUR SOLUTION
        zmax= array_api.max(Z, axis=1, keepdims=True)
        expz=array_api.exp(Z-zmax)
        logsumexp=array_api.log(array_api.sum(expz, axis=1))+ array_api.max(Z, axis=1)
        logsoft=Z-array_api.reshape(logsumexp,(-1,1))
        return logsoft


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        Z = node.inputs[0]
        # print(Z.shape)
        # maxz=array_api.max(Z,axis=1, keepdims=True)
        # print(maxz.shape)
        # expz= exp(Z-maxz)
        # print(expz.shape)
        # sumexpz=summation(expz,axes=1).reshape((-1,1)).broadcast_to(Z.shape)
        # print(sumexpz)
        out_grad_row_sum = summation(out_grad, axes=(1,)).reshape((-1, 1)).broadcast_to(Z.shape)
        # grad_input_log_sum_exp = out_grad_row_sum / sumexpz
        # grad_input_log_sum_exp = grad_input_log_sum_exp * expz
        g=exp(node)
        return out_grad -( out_grad_row_sum*g)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        # self.axes = axes
        self.axes = axes
        if isinstance(axes, int):
            self.axes = tuple([axes])

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # zmax= array_api.max(Z, axis=self.axes, keepdims=True)
        # expz=array_api.exp(Z-zmax)
        # logsumexp=array_api.log(array_api.sum(expz, axis=self.axes))+ array_api.max(Z, axis=self.axes, keepdims=False)
        # return logsumexp
        Z_max = Z.max(axis=self.axes)
        Z_shape = list(Z.shape)
        if self.axes is not None:
            for axis in self.axes:
                Z_shape[axis] = 1
            Z_max_reshaped = Z_max.reshape(tuple(Z_shape))
        else:
            Z_max_reshaped = Z_max.reshape(tuple([1 for _ in Z_shape]))
        Z_normalized = Z - Z_max_reshaped.broadcast_to(Z.shape)
        return array_api.log(array_api.summation( array_api.exp(Z_normalized), axis = self.axes )) + Z_max
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Z = node.inputs[0]
        # zmax = array_api.max(Z, axis=self.axes, keepdims=True)
        # expz = array_api.exp(Z - zmax)
        # sum_expz = array_api.sum(expz, axis=self.axes, keepdims=True)
        # softmax = expz / sum_expz
        
        # # Multiply the softmax by the incoming gradient
        # return out_grad * softmax
        # Z = node.inputs[0]
        # if self.axes:
        #   shape = [1] * len(Z.shape)
        #   j = 0
        #   for i in range(len(shape)):
        #     if i not in self.axes:
        #       shape[i] = node.shape[j]
        #       j += 1
        #   node_new = node.reshape(shape)
        #   grad_new = out_grad.reshape(shape)
        # else:
        #   node_new = node
        #   grad_new = out_grad
        # return grad_new * exp(Z - node_new)
        Z = node.inputs[0]
        Z_max = Tensor(Z.numpy().max(axis = self.axes), device = Z.device)

        Z_shape_for_reshape = list(Z.shape)
        if self.axes is not None:
            for axis in self.axes:
                Z_shape_for_reshape[axis] = 1
        else:
            for i in range(len(Z_shape_for_reshape)):
                Z_shape_for_reshape[i] = 1
        Z_shape_for_reshape = tuple(Z_shape_for_reshape)
        Z_shape_for_broadcast = Z.shape

        Z_max_reshaped_broadcasted = broadcast_to(reshape(Z_max, Z_shape_for_reshape), Z_shape_for_broadcast)
        Z_minus_Z_max = Z - Z_max_reshaped_broadcasted
        Z_exp = exp(Z_minus_Z_max)
        Z_sum_exp = broadcast_to(reshape(summation(Z_exp, self.axes), Z_shape_for_reshape), Z_shape_for_broadcast)
        return multiply(broadcast_to(reshape(out_grad, Z_shape_for_reshape), Z_shape_for_broadcast), divide(Z_exp, Z_sum_exp))
 
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

