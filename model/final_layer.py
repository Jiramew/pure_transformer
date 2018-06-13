import numpy as np
from model.layer import Layer


class FinalLayer(Layer):
    def __init__(self,
                 name,
                 model_dimension,
                 word_count,
                 network):
        super(FinalLayer, self).__init__(name=name)
        self.weight = 0.0005 * np.random.randn(model_dimension, word_count)

        self.inputs = None
        self.outputs = None

        self.back_error = None
        self.weight_grad = None

        self.network = network

    def forward(self):
        self.inputs = self.pre_layer.outputs

        scores = np.dot(self.inputs, self.weight)
        exp_scores = np.exp(scores)
        outputs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        self.outputs = outputs
        return outputs

    def backward(self):
        error = self.outputs
        for j in range(self.network.batch_size):
            error[j][range(self.network.max_len), self.network.labels[j]] -= 1

        back_error = np.dot(error, np.transpose(self.weight))
        weight_grad = np.array(
            [np.dot(np.transpose(self.inputs[j, :, :]), error[j, :, :]) for j in range(self.network.batch_size)])
        weight_grad = np.sum(weight_grad, axis=0) / self.network.batch_size

        self.back_error = back_error
        self.weight_grad = weight_grad

    def batch_update(self):
        self.weight += -1.0 * self.network.learning_rate * self.weight_grad
