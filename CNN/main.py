from layers import *
class CNN:
    def __init__(self):
        lr = 0.01
        self.layers = []
        self.layers.append(
            ConvolutionLayer(inputs_channel=1, num_filters=6, width=5, height=5, padding=2, stride=1, learning_rate=lr,
                          name='conv1'))
        self.layers.append(ReLu())
        self.layers.append(MaxPoolingLayer(width=2, height=2, stride=2, name='maxpool2'))
        self.layers.append(
            ConvolutionLayer(inputs_channel=6, num_filters=16, width=5, height=5, padding=0, stride=1, learning_rate=lr,
                          name='conv3'))
        self.layers.append(ReLu())
        self.layers.append(MaxPoolingLayer(width=2, height=2, stride=2, name='maxpool4'))
        self.layers.append(
            ConvolutionLayer(inputs_channel=16, num_filters=120, width=5, height=5, padding=0, stride=1, learning_rate=lr,
                          name='conv5'))
        self.layers.append(ReLu())
        self.layers.append(Flatten())
        self.layers.append(FullyConnectedLayer(num_inputs=120, num_outputs=84, learning_rate=lr, name='fc6'))
        self.layers.append(ReLu())
        self.layers.append(FullyConnectedLayer
                           (num_inputs=84, num_outputs=10, learning_rate=lr, name='fc7'))
        self.layers.append(Softmax())
        self.lay_num = len(self.layers)