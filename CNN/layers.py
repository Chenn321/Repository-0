import numpy as np


class FullyConnectedLayer:

    def __init__(self, in_features, out_features, learning_rate):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = 0.01 * np.random.rand(in_features, out_features)
        self.bias = np.zeros(self.out_features)
        self.lr = learning_rate

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias

    def backward(self, dy):

        dx = np.dot(dy, self.weights.T)
        dw = np.dot(self.inputs.T, dy)
        db = np.sum(dy, axis=0)
        #db = dy

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        return dx
    
class ConvolutionLayer:
    def __init__(self, in_channels, num_kernels, width, height, stride, padding, learning_rate):

        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.width = width
        self.height = height
        self.stride = stride
        self.padding = padding
        self.lr = learning_rate

        self.weights = np.random.randn((self.num_kernels, self.in_channels, self.width, self.height))*0.01
        self.bias = np.zeros(self.num_kernels)


    def forward(self, inputs):
        c, w, h = inputs.shape
        ww = w + 2 * self.padding
        hh = h + 2 * self.padding

        padding_inputs = np.zeros((c, ww, hh))
        
        for i in range(c):
            padding_inputs[c, self.padding : w + self.padding, self.padding : h + self.padding] = inputs
        self.inputs = padding_inputs

        w_out = (ww - self.width) // self.stride + 1
        h_out = (hh - self.height) // self.stride + 1

        outputs = np.zeros((self.num_kernels, w_out, h_out))
        for i in range(self.num_kernels):
            for j in range(w_out):
                for k in range(h_out):
                    inputs_block = self.inputs[:, j : j + self.width, k : k + self.height]
                    outputs[i,j,k] = np.sum(inputs_block * self.weights[i, :, :, :]) + self.bias[i]

        return outputs

    def backward(self, dy):
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        f, w, h = dy.shape
        for i in range(f):
            for j in range(w):
                for k in range(h):
                    dw[i,:,:,:] += dy[i,j,k] * self.inputs[:, j: j+self.width, k: k+self.height]
                    dx[:, j: j+self.width, k: k+self.height] += dy[i,j,k] * self.weights[i,:,:,:]

        for i in range(f):
            db[i] = np.sum(dy[i, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        return dx
    
class Flatten:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.c, self.w, self.h = inputs.shape
        return inputs.reshape(1, self.c * self.w * self.h)
    
    def backward(self, dy):
        return dy.reshape(self.c, self.w, self.h)
    
def cross_entropy(inputs, labels):
    out_num = labels.shape[0]
    p = np.sum(labels.reshape(1,out_num) * inputs)
    loss = -np.log(p)

    return loss

class MaxPoolingLayer:
    def __init__(self, width, height, stride, name):
        self.width = width
        self.height = height
        self.stride = stride
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        c, w, h = inputs.shape
        w_out = (w - self.width) // self.stride + 1
        h_out = (h - self.height) // self.stride + 1
        out = np.zeros((c, w_out, h_out))

        for i in range(c):
            for j in range(w_out):
                for k in range(h_out):
                    out[i, j, k] = np.max(
                        self.inputs[i, j*self.stride: j*self.stride+self.width, k*self.stride: k*self.stride+self.height])
        return out

    def backward(self, dy):
        c, w, h = self.inputs.shape
        dx = np.zeros(self.inputs.shape)

        for i in range(c):
            for j in range(0, w, self.width):
                for k in range(0, h, self.height):
                    st = np.argmax(self.inputs[i, j:j + self.width, k:k + self.height])
                    (idx, idy) = np.unravel_index(st, (self.width, self.height))
                    dx[i, j + idx, k + idy] = dy[i, j // self.width, k // self.height]
        return dx
    
class ReLu:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        out = inputs.copy()
        out[out < 0] = 0
        return out
    
    def backward(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx
    
class Softmax:
    def __init__(self):
        pass

    def forward(self, inputs):
        exp = np.exp(inputs, dtype=np.float)
        self.out = exp / np.sum(exp)
        return self.out
    
    def backward(self, dy):
        return self.out.T - dy.reshape(dy.shape[0], 1)