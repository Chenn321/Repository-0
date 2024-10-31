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

        # 所有卷积参数构成一个3维矩阵， (num_filters, channel, width, height)

        self.weights = np.random.randn((self.num_kernels, self.in_channels, self.width, self.height))*0.01
        self.bias = np.zeros(self.num_kernels)
        # for i in range(self.num_filters):
        #     self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.channel*self.width*self.height)), size=(self.channel, self.width, self.height))


    # def zero_padding(self, inputs, padding_size):
    #     w, h = inputs.shape[0], inputs.shape[1]
    #     new_w = 2 * padding_size + w
    #     new_h = 2 * padding_size + h
    #     out = np.zeros((new_w, new_h))
    #     out[padding_size:w+padding_size, padding_size:h+padding_size] = inputs
    #     return out

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

        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:,w:w+self.width,h:h+self.height]
                    dx[:,w:w+self.width,h:h+self.height]+=dy[f,w,h]*self.weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx
    
class Flatten:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.C, self.W, self.H = inputs.shape
        return inputs.reshape(1, self.C*self.W*self.H)
    def backward(self, dy):
        return dy.reshape(self.C, self.W, self.H)
    
def cross_entropy(inputs, labels):

    out_num = labels.shape[0]
    p = np.sum(labels.reshape(1,out_num)*inputs)
    loss = -np.log(p)
    return loss

class MaxPoolingLayer:
    # A Max Pooling layer .
    def __init__(self, width, height, stride, name):
        self.width = width
        self.height = height
        self.stride = stride
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = (W - self.width) // self.stride + 1
        new_height = (H - self.height) // self.stride + 1
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(new_width):
                for h in range(new_height):
                    out[c, w, h] = np.max(
                        self.inputs[c, w * self.stride:w * self.stride + self.width, h * self.stride:h * self.stride + self.height])
        return out

    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)

        for c in range(C):
            for w in range(0, W, self.width):
                for h in range(0, H, self.height):
                    st = np.argmax(self.inputs[c, w:w + self.width, h:h + self.height])
                    (idx, idy) = np.unravel_index(st, (self.width, self.height))
                    dx[c, w + idx, h + idy] = dy[c, w // self.width, h // self.height]
        return dx
    
class ReLu:
    def __init__(self):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret
    def backward(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx
    
class Softmax:
    def __init__(self):
        pass
    def forward(self, inputs):
        exp = np.exp(inputs, dtype=np.float)
        self.out = exp/np.sum(exp)
        return self.out
    def backward(self, dy):
        return self.out.T - dy.reshape(dy.shape[0],1)