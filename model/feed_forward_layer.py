import numpy as np
from model.layer import Layer


class FeedForwardLayer(Layer):
    def __init__(self,
                 name,
                 batch_size,
                 dimension_inner,
                 model_dimension,
                 network):
        super(FeedForwardLayer, self).__init__(name=name)
        self.batch_size = batch_size
        self.dimension_inner = dimension_inner
        self.model_dimension = model_dimension

        self.feed_weight1 = 0.0005 * np.random.randn(1, dimension_inner)
        self.feed_weight2 = 0.0005 * np.random.randn(1, model_dimension)

        self.inputs = None
        self.outputs = None

        self.relu_index = None

        self.network = network

        self.back_error = None
        self.feed_weight1_grad = None
        self.feed_weight2_grad = None

    def forward(self):
        self.inputs = self.pre_layer.outputs
        self.inputs[self.inputs < 0] = 0

        z1 = np.dot(self.inputs, np.tile(self.feed_weight1, (self.model_dimension, 1)))
        self.relu_index = (z1 < 0)
        z1[self.relu_index] = 0

        z2 = np.dot(z1, np.tile(self.feed_weight2, (self.dimension_inner, 1)))
        sum_of_z = z2 + self.inputs

        self.outputs = sum_of_z
        return sum_of_z

    def backward(self):
        diff = self.next_layer.back_error
        z2_diff = np.dot(diff, np.transpose(np.tile(self.feed_weight2, (self.dimension_inner, 1))))

        feed_weight2_grad = np.array(
            [np.dot(np.transpose(self.outputs[i, :, :]), diff[i, :, :]) for i in range(self.batch_size)])
        feed_weight2_grad = np.sum(feed_weight2_grad, axis=(0, 1)) / (self.dimension_inner * self.batch_size)

        z1_diff = z2_diff
        z1_diff[self.relu_index] = 0

        feed_weight1_grad = np.array(
            [np.dot(np.transpose(self.inputs[i, :, :]), z1_diff[i, :, :]) for i in range(self.batch_size)])
        feed_weight1_grad = np.sum(feed_weight1_grad, axis=(0, 1)) / (self.model_dimension * self.batch_size)

        dinputs = np.dot(z1_diff, np.transpose(np.tile(self.feed_weight1, (self.model_dimension, 1))))
        dinputs[self.inputs < 0] = 0
        dinputs = dinputs + self.next_layer.back_error

        self.feed_weight1_grad = feed_weight1_grad.reshape(1, self.dimension_inner)
        self.feed_weight2_grad = feed_weight2_grad.reshape(1, self.model_dimension)
        self.back_error = dinputs

    def batch_update(self):
        self.feed_weight1 += -1 * self.network.learning_rate * self.feed_weight1_grad
        self.feed_weight2 += -1 * self.network.learning_rate * self.feed_weight2_grad
