import numpy as np
from model.layer import Layer


class PositionalEncodingLayer(Layer):
    def __init__(self,
                 name,
                 zero_padding,
                 scale,
                 model_dimension,
                 network):
        super(PositionalEncodingLayer, self).__init__(name=name)
        self.zero_padding = zero_padding
        self.scale = scale

        self.model_dimension = model_dimension

        self.inputs = None
        self.outputs = None

        self.network = network

    def forward(self,
                inputs):
        self.inputs = inputs

        batch_size, dim = inputs.shape
        natural_order = np.tile(np.array(range(dim)), (batch_size, 1))

        pe = np.array([[pos / np.power(10000, 2. * i / self.model_dimension) for i in range(self.model_dimension)]
                       for pos in range(dim)])

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        position_weight = 0.0005 * pe

        if self.zero_padding:
            position_weight = np.concatenate((np.zeros((1, self.model_dimension), dtype=int),
                                              position_weight[1:, :]),
                                             axis=0)
        outputs = position_weight[natural_order]

        if self.scale:
            outputs = outputs * (self.model_dimension ** 0.5)

        self.outputs = outputs
        return self.outputs

    def backward(self):
        pass

    def batch_update(self):
        pass
