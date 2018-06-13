import numpy as np
from model.layer import Layer


class MultiheadAttentionLayer(Layer):
    def __init__(self,
                 name,
                 batch_size,
                 model_dimension,
                 network):
        super(MultiheadAttentionLayer, self).__init__(name=name)
        self.batch_size = batch_size

        self.attention_weight = 0.0005 * np.random.randn(4, model_dimension, model_dimension)

        self.is_with_encoder_outputs = False
        self.with_encoder_outputs = None

        self.inputs = None
        self.outputs = None

        self.network = network

        self.scale = None
        self.softmax = None
        self.matmul_withv = None
        self.q = None
        self.k = None
        self.v = None
        self.query = None
        self.key = None

        self.back_error = None
        self.attention_weight_grad = np.zeros(self.attention_weight.shape)

    def set_encoder_outputs(self, outputs):
        self.is_with_encoder_outputs = True
        self.with_encoder_outputs = outputs

    def forward(self):
        self.inputs = self.pre_layer.outputs

        if self.is_with_encoder_outputs:
            query = self.inputs
            key = self.with_encoder_outputs
        else:
            query = self.inputs
            key = self.inputs

        q = np.dot(query, self.attention_weight[0, :, :])
        k = np.dot(key, self.attention_weight[1, :, :])
        v = np.dot(key, self.attention_weight[2, :, :])

        matmul_qk = np.array([np.dot(q[i, :, :], np.transpose(k[i, :, :])) for i in range(self.batch_size)])

        scale = matmul_qk / (k.shape[2] ** 0.5)
        scale[scale == 0] = -2 ** 32 + 1

        softmax_exp = np.exp(scale)
        softmax_z = np.sum(softmax_exp, axis=2)
        softmax = np.array([softmax_exp[i, :, :] / softmax_z[i, :].reshape(10, 1) for i in range(self.batch_size)])
        matmul_withv = np.array([np.dot(softmax[i, :, :], v[i, :, :]) for i in range(self.batch_size)])
        dense = np.array(
            [np.dot(matmul_withv[i, :, :], self.attention_weight[3, :, :]) for i in range(self.batch_size)])

        outputs = dense + query

        self.outputs = outputs
        self.scale = scale
        self.softmax = softmax
        self.matmul_withv = matmul_withv
        self.q = q
        self.k = k
        self.v = v
        self.query = query
        self.key = key

    def backward(self):
        error = self.next_layer.back_error
        matmul_withv_diff = np.array(
            [np.dot(error[i, :, :], np.transpose(self.attention_weight[3, :, :])) for i in range(self.batch_size)])

        v_diff = np.array(
            [np.dot(np.transpose(self.softmax[i, :, :]), matmul_withv_diff[i, :, :]) for i in range(self.batch_size)])

        softmax_diff = np.array(
            [np.dot(matmul_withv_diff[i, :, :], np.transpose(self.v[i, :, :])) for i in range(self.batch_size)])
        softmax_diff[self.scale == 0] = 0
        matmul_qk = softmax_diff * (self.k.shape[2] ** 0.5)

        q_diff = np.array([np.dot(matmul_qk[i, :, :], self.k[i, :, :]) for i in range(self.batch_size)])
        k_diff = np.array(
            [np.transpose(np.dot(np.transpose(self.q[i, :, :]), matmul_qk[i, :, :])) for i in range(self.batch_size)])

        query_diff = np.dot(q_diff, np.transpose(self.attention_weight[0, :, :])) + self.outputs
        key_diff = np.dot(k_diff, self.attention_weight[1, :, :]) + np.dot(v_diff, self.attention_weight[2, :, :])

        weight_0_grad = np.array(
            [np.dot(np.transpose(self.query[i, :, :]), self.q[i, :, :]) for i in
             range(self.batch_size)]) / self.batch_size
        self.attention_weight_grad[0, :, :] = np.sum(weight_0_grad, axis=0)
        weight_1_grad = np.array(
            [np.dot(np.transpose(self.key[i, :, :]), self.k[i, :, :]) for i in
             range(self.batch_size)]) / self.batch_size
        self.attention_weight_grad[1, :, :] = np.sum(weight_1_grad, axis=0)
        weight_2_grad = np.array(
            [np.dot(np.transpose(self.key[i, :, :]), self.v[i, :, :]) for i in
             range(self.batch_size)]) / self.batch_size
        self.attention_weight_grad[2, :, :] = np.sum(weight_2_grad, axis=0)
        weight_3_grad = np.array(
            [np.dot(np.transpose(self.matmul_withv[i, :, :]), error[i, :, :]) for i in
             range(self.batch_size)]) / self.batch_size
        self.attention_weight_grad[3, :, :] = np.sum(weight_3_grad, axis=0)

        if self.is_with_encoder_outputs:
            self.back_error = query_diff
            self.network.decoder_error = key_diff / 2
        else:
            self.back_error = (query_diff + key_diff) / 3

    def batch_update(self):
        self.attention_weight += -1 * self.network.learning_rate * self.attention_weight_grad
