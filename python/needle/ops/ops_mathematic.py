"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return (out_grad * self.scalar * (lhs ** (self.scalar - 1)),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (
            power_scalar(rhs, -1) * out_grad,
            -1 * lhs * power_scalar(rhs, -2) * out_grad,
        )


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
            return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad * (1 / self.scalar),)



def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
    #     index = list(range(len(a.shape)))
    #     if self.axes is None:
    #         index[-1], index[-2] = index[-2], index[-1]
    #     else:
    #         axis1 = self.axes[0]
    #         axis2 = self.axes[1]
    #         index[axis1], index[axis2] = index[axis2], index[axis1]
    #     return a.permute(tuple(index))
    # def gradient(self, out_grad, node):
    #     a = node.inputs[0]
    #     return (transpose(out_grad, self.axes),)
    

        ### BEGIN YOUR SOLUTION
        # print(a.device)
        if self.axes:
            ax0, ax1 = self.axes[0], self.axes[1]
        else:
            ax0, ax1 = a.ndim - 2, a.ndim - 1
        permute_axes = list(range(a.ndim))
        permute_axes[ax0], permute_axes[ax1] = ax1, ax0
        return a.permute(permute_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out=out_grad.transpose(self.axes)
        return out


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


# class BroadcastTo(TensorOp):
#     def __init__(self, shape):
#         self.shape = shape

#     def compute(self, a):
#         return array_api.broadcast_to(a, self.shape)

#     def gradient(self, out_grad, node):
#         a = node.inputs[0]
#         return (reverse_broadcast(out_grad, a.shape),)




class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # a: (1,3) -> (1,1,3,3)
        a_shape = node.inputs[0].shape
        shape = [1] * (len(self.shape) - len(a_shape)) + list(a_shape)
        dele_shape = []
        for i in range(len(self.shape)):
            if self.shape[i] != shape[i]:
                dele_shape.append(i)
        return reshape(summation(out_grad, tuple(dele_shape)), a_shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return a.sum(axis = None)
        elif isinstance(self.axes, int) or (isinstance(self.axes, (list, tuple)) and len(self.axes) == 1):
            return a.sum(self.axes)
        else:
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a

    def gradient(self, out_grad, node):
        # a = node.inputs[0]

        # if self.axes is None:
        #     return broadcast_to(out_grad, a.shape)

        # out_shape = list(out_grad.shape)
        # for axis in self.axes:
        #     out_shape.insert(axis, 1)

        # return broadcast_to(reshape(out_grad, out_shape), a.shape)
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError("Unsupported axes type, must be int, tuple or None!")
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)

def reverse_broadcast(out, a_shape):
    out_shape = list(out.shape)

    a_ind = 0
    out_ind = 0

    diff = len(out_shape) - len(a_shape)
    a_ind -= diff

    axis = 0
    while a_ind < len(a_shape) and out_ind < len(out_shape):
        if a_ind < 0:
            n = out_shape[0]
            out = summation(out, axes=(axis,))
            out_shape.pop(0)

            a_ind += 1
            continue

        if a_shape[a_ind] == out_shape[out_ind]:
            a_ind += 1
            out_ind += 1
            axis += 1
            continue

        n = out_shape[out_ind]
        out_shape[out_ind] = 1
        out = reshape(summation(out, axes=(axis,)), out_shape)
        a_ind += 1
        out_ind += 1

    return out

# class MatMul(TensorOp):
#     def compute(self, a, b):
#         return a @ b

#     def gradient(self, out_grad, node):
#         lhs, rhs = node.inputs

#         if lhs.shape == rhs.shape:
#             return (matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad))

#         l = reduce(operator.mul, lhs.shape[:-2], 1)
#         r = reduce(operator.mul, rhs.shape[:-2], 1)

#         if l < r:
#             return reverse_broadcast(
#                 matmul(out_grad, transpose(rhs)), lhs.shape
#             ), matmul(transpose(lhs), out_grad)
#         else:
#             return matmul(out_grad, transpose(rhs)), reverse_broadcast(
#                 matmul(transpose(lhs), out_grad), rhs.shape
#             )
class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        # (3, 2, 3) (3, 2) -> (3, 2, 2)
        # lhs, rhs = node.inputs
        # if len(lhs.shape) == len(rhs.shape):
        #     return out_grad @ transpose(rhs), transpose(lhs) @ out_grad
        # elif len(lhs.shape) > len(rhs.shape):
        #     out = transpose(lhs) @ out_grad
        #     for _ in range(len(lhs.shape) - len(rhs.shape)):
        #         out = summation(out, 0)
        #     return out_grad @ transpose(rhs), out
        # else:
        #     out = out_grad @ transpose(rhs)
        #     for _ in range(len(rhs.shape) - len(lhs.shape)):
        #         out = summation(out, 0)
        #     return out, transpose(lhs) @ out_grad
        lhs, rhs = node.inputs
        print("rhs",rhs.device)
        print("out_grad", out_grad.device)
        lgrad= matmul(out_grad, rhs.transpose())
        print("lgrad",lgrad.device)
        rgrad = matmul(lhs.transpose(), out_grad)
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
        ## END YOUR SOLUTION
        
# class MatMul(TensorOp):
#     def compute(self, a, b):
#         ### BEGIN YOUR SOLUTION
#         # return array_api.matmul(a,b)
#         return a @ b
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         left, right = node.inputs
#         grad_left, grad_right = matmul(out_grad,transpose(right)), matmul(transpose(left),out_grad)
#         grad_left_summed = summation(grad_left, tuple(range(len(grad_left.shape) - len(left.shape)))) #summed batch
#         grad_right_summed = summation(grad_right, tuple(range(len(grad_right.shape) - len(right.shape)))) #summed batch
#         print("matmul", grad_left_summed.device,grad_right_summed.device )
#         return grad_left_summed, grad_right_summed
#         ### END YOUR SOLUTION




def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -1 * a

    def gradient(self, out_grad, node):
        return -1 * out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return a.log()

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (power_scalar(a, -1) * out_grad,)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return a.exp()

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return exp(a) * out_grad



def exp(a):
    return Exp()(a)



class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a,0)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        mask = node.cached_data
        mask =  mask / (mask + 1e-10)
        grad=out_grad*mask
        print("Relu:", grad.device)
        return grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad=out_grad*(1-node.cached_data*node.cached_data)
        print("tanh:", grad.device)
        return grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
       
        shap = list(args[0].shape)
      
        shap.insert(self.axis, len(args))

        
        result = array_api.empty(shape=shap, device=args[0].device)
        idxs = []
        for sh in args[0].shape:
            idxs.append(slice(0, sh, 1))
        for i in range(len(args)):
            new_idxs = idxs.copy()
            new_idxs.insert(self.axis, i)
            result[tuple(new_idxs)] = args[i]
        
        return result
      

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad=split(out_grad, self.axis)
        print("stack:", grad.device)
        return grad
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # shape = A.shape
        # num_splits = shape[self.axis]
        
        # # Split `A` along the specified axis
        # split_tensors = [A.take(i, axis=self.axis) for i in range(num_splits)]
        
        # return tuple(split_tensors)
        ndim = A.shape[self.axis]
        idxs = []
        for i, sh in enumerate(A.shape):
            if i != self.axis:
                idxs.append(slice(0, sh, 1))
        ret = []
        for i in range(ndim):
            new_idxs = idxs.copy()
            new_idxs.insert(self.axis, i)
            it = A[tuple(new_idxs)].compact()
            ret.append(it.sum(self.axis))
        return tuple(ret)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad=stack(out_grad, self.axis)
        print("Split:", grad.device)
        return grad
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)

        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        valid_axes = []
        for axis in self.axes:
            if 0 <= axis < len(a.shape):
                valid_axes.append(axis)
        new_shape = list(a.shape)
        for axis in valid_axes:
            new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        new_array = array_api.full(tuple(new_shape), 0, device=a.device)
        slices = [slice(0, shape) for shape in new_shape]
        for axis in valid_axes:
            slices[axis] = slice(0, new_shape[axis], self.dilation + 1)
        new_array[tuple(slices)] = a
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad=undilate(out_grad, self.axes, self.dilation)
        print("dilate", grad.device)
        return grad
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = self.axes
        if isinstance(axes, int):
            axes = (axes,)
        elif axes is None:
            axes = tuple(range(len(a.shape)))
        
        # Normalize negative axes
        axes = tuple(axis if axis >= 0 else axis + len(a.shape) for axis in axes)
        
        # Filter out-of-bounds axes
        valid_axes = [axis for axis in axes if 0 <= axis < len(a.shape)]
        # valid_axes = []
        # for axis in self.axes:
        #     if 0 <= axis < len(a.shape):
        #         valid_axes.append(axis)
        slices = [slice(0, shape) for shape in a.shape]
        for axis in valid_axes:
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        undilated= a[tuple(slices)]
        return undilated
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad=dilate(out_grad, self.axes, self.dilation)
        print("Undilate:", grad.device)
        return grad
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)



class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        # print("A:", A.shape)
        K, _, _, C_out = B.shape
        # print("B:", B.shape)
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * C_in
        out_dim_h=(H - K + 1) // self.stride 
        
        out_dim_w=(W - K + 1) // self.stride 
        # print("out dim:", (N, out_dim_h,  out_dim_w, K, K, C_in))
        strided_A = A.as_strided(
            shape=(N, out_dim_h,  out_dim_w, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact()
        # print("Strided_A",strided_A.compact().shape)
        # print("Innder_dim",inner_dim)
        # reshape_B=B.compact().reshape((-1, C_out))
        strided_A=strided_A.reshape((N * out_dim_h * out_dim_w, inner_dim))
        # print("passing")
        reshape_B=B.compact().reshape((inner_dim, C_out))
        out = strided_A @ reshape_B
        return out.compact().reshape((N, out_dim_h, out_dim_w, C_out))

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X = node.inputs[0]
        W = node.inputs[1]
        K = W.shape[0]
        if self.stride > 1:
            # N, (H + 2P - K + 1) // self.stride, (W + 2P - K + 1) // self.stride, C_out
            out_grad = dilate(out_grad, (1, 2), self.stride-1) # N, (H + 2P - K + 1), (W + 2P - K + 1), C_out
        W_flip = flip(W, (0, 1)) # K, K, C_in, C_out
        W_transpose = transpose(W_flip, (2, 3)) # K, K, C_out, C_in
        X_grad = conv(out_grad, W_transpose, padding=K-1-self.padding)

        X_permute = transpose(X, (0, 3))
        out_grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))
        W_grad_transpose = conv(X_permute, out_grad_permute, padding=self.padding)
        W_grad = transpose(transpose(W_grad_transpose, (0, 1)), (1, 2))
        print("conv:", X_grad.device, W_grad.device)
        return X_grad, W_grad
        ### END YOUR SOLUTION
    
# conv compute was giving error before when reshape was like(-1,innerdim), changing all values to postive made it right


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


# i was having issue with backwards, changed code for matmul( as it was using reversed broadcast) and broadcast to