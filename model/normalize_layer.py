import numpy as np
from model.layer import Layer


class NormalizeLayer(Layer):
    def __init__(self,
                 name,
                 network):
        super(NormalizeLayer, self).__init__(name=name)

        self.inputs = None
        self.outputs = None

        self.mean = None
        self.variance = None

        self.network = network

        self.back_error = None

    def forward(self):
        self.inputs = self.pre_layer.outputs

        mean = np.mean(self.inputs, axis=-1, keepdims=True)
        variance = np.std(self.inputs, axis=-1, keepdims=True)
        outputs = (self.inputs - mean) / ((variance + 1e-8) ** 0.5)

        self.mean = mean
        self.variance = variance
        self.outputs = outputs

        return outputs

    def backward(self):
        error = self.mean + ((self.variance + 1e-8) ** .5) * self.next_layer.back_error
        self.back_error = error

    def batch_update(self):
        pass
