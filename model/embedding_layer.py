import numpy as np
from model.layer import Layer


class EmbeddingLayer(Layer):
    def __init__(self,
                 name,
                 zero_padding,
                 scale,
                 word_count,
                 model_dimension,
                 network):
        super(EmbeddingLayer, self).__init__(name=name)
        self.zero_padding = zero_padding
        self.scale = scale

        self.model_dimension = model_dimension

        self.word_table_weight = 0.0005 * np.random.randn(word_count, model_dimension)

        self.inputs = None
        self.outputs = None

        self.network = network

        self.back_error = None
        self.word_table_weight_grad = None

    def forward(self,
                inputs):
        self.inputs = inputs

        if self.zero_padding:
            self.word_table_weight = np.concatenate(
                (
                    np.zeros(
                        (1, self.model_dimension),
                        dtype=int
                    ),
                    self.word_table_weight[1:, :]
                ),
                axis=0
            )
        outputs = self.word_table_weight[inputs]

        if self.scale:
            outputs = outputs * (self.model_dimension ** 0.5)

        self.outputs = outputs
        return self.outputs

    def backward(self):
        self.back_error = self.network.decoder_error
        self.word_table_weight_grad = self.next_layer.back_error / (self.model_dimension ** 0.5)

    def batch_update(self):
        self.word_table_weight_grad += -1 * self.network.learning_rate * self.word_table_weight_grad
