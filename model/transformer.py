import numpy as np
from model.embedding_layer import EmbeddingLayer
from model.positional_encoding import PositionalEncodingLayer
from model.addition_layer import AdditionLayer
from model.multihead_attention_layer import MultiheadAttentionLayer
from model.feed_forward_layer import FeedForwardLayer
from model.normalize_layer import NormalizeLayer
from model.final_layer import FinalLayer


class Transformer(object):
    def __init__(self, batch_size, source_word_count, target_word_count, max_len):
        self.batch_size = batch_size

        self.layers = []
        self.next_layer = []
        self.next_layer_index = 0

        self.learning_rate = 0.01
        self.regularization = 0.1
        self.max_len = max_len

        self.model_dimension = 512
        self.dimension_inner = 2048

        self.source_word_count = source_word_count
        self.target_word_count = target_word_count

        self.encoder_outputs = None
        self.decoder_error = None

        self.features = None
        self.labels = None

        self._model()

    def add_layer(self, new_layer):
        if self.next_layer_index > 0:
            pre_layer = self.layers[self.next_layer_index - 1]
            pre_layer.set_output_layer(new_layer)
            new_layer.set_input_layer(pre_layer)

        new_layer.layer_index = self.next_layer_index
        self.layers.append(new_layer)
        self.next_layer_index += 1

    def _model(self):
        # Input
        embedding = EmbeddingLayer(name="Input-embedding-0",
                                   zero_padding=True,
                                   scale=True,
                                   word_count=self.source_word_count,
                                   model_dimension=self.model_dimension,
                                   network=self)
        positional_encoder = PositionalEncodingLayer(name="Input-positional_encoder-0",
                                                     zero_padding=True,
                                                     scale=True,
                                                     model_dimension=self.model_dimension,
                                                     network=self)
        add_embdeding_postion = AdditionLayer(name="Input-add-0",
                                              input_list=[embedding, positional_encoder],
                                              network=self)
        self.add_layer(add_embdeding_postion)

        # Encoder
        for i in range(1, 2):
            self.add_layer(
                MultiheadAttentionLayer(name="Encoder-multihead_attention_1-{0}".format(i),
                                        batch_size=self.batch_size,
                                        model_dimension=self.model_dimension,
                                        network=self)
            )
            self.add_layer(NormalizeLayer(name="Encoder-normalize_1-{0}".format(i),
                                          network=self))
            self.add_layer(
                FeedForwardLayer(name="Encoder-feedforward_1-{0}".format(i),
                                 batch_size=self.batch_size,
                                 dimension_inner=self.dimension_inner,
                                 model_dimension=self.model_dimension,
                                 network=self)
            )
            self.add_layer(NormalizeLayer(name="Encoder-normalize_2-{0}".format(i),
                                          network=self))

        # Output
        embedding = EmbeddingLayer(name="Output-embedding-0",
                                   zero_padding=True,
                                   scale=True,
                                   word_count=self.target_word_count,
                                   model_dimension=self.model_dimension,
                                   network=self)
        positional_encoder = PositionalEncodingLayer(name="Output-positional_encoder-0",
                                                     zero_padding=True,
                                                     scale=True,
                                                     model_dimension=self.model_dimension,
                                                     network=self)
        add_embdeding_postion = AdditionLayer(name="Output-add-0",
                                              input_list=[embedding, positional_encoder],
                                              network=self)
        self.add_layer(add_embdeding_postion)

        # Decoder
        for j in range(1, 2):
            self.add_layer(
                MultiheadAttentionLayer(name="Decoder-multihead_attention_1-{0}".format(j),
                                        batch_size=self.batch_size,
                                        model_dimension=self.model_dimension,
                                        network=self)
            )
            self.add_layer(NormalizeLayer(name="Decoder-normalize_1-{0}".format(j),
                                          network=self))
            self.add_layer(
                MultiheadAttentionLayer(name="Decoder-multihead_attention_2-{0}".format(j),
                                        batch_size=self.batch_size,
                                        model_dimension=self.model_dimension,
                                        network=self),
            )
            self.add_layer(NormalizeLayer(name="Decoder-normalize_2-{0}".format(j),
                                          network=self))
            self.add_layer(
                FeedForwardLayer(name="Decoder-feedforward_1-{0}".format(j),
                                 batch_size=self.batch_size,
                                 dimension_inner=self.dimension_inner,
                                 model_dimension=self.model_dimension,
                                 network=self)
            )
            self.add_layer(NormalizeLayer(name="Decoder-normalize_3-{0}".format(j),
                                          network=self))

        # Final Output
        self.add_layer(FinalLayer(name="Final",
                                  model_dimension=self.model_dimension,
                                  word_count=self.target_word_count,
                                  network=self))

    def _forward(self, data_in, data_out):
        self.features = data_in
        self.labels = data_out

        for i in range(len(self.layers)):
            if self.layers[i].name == "Input-add-0":
                self.layers[i].forward(self.features)
            elif self.layers[i].name == "Output-add-0":
                self.encoder_outputs = self.layers[i].pre_layer.outputs
                outputs = np.concatenate((np.ones((self.batch_size, 1), dtype=int) * 2, self.labels[:, :-1]), axis=1)
                self.layers[i].forward(outputs)
            else:
                if "Decoder-multihead_attention_2" in self.layers[i].name:
                    self.layers[i].set_encoder_outputs(self.encoder_outputs)
                self.layers[i].forward()

    def _backward(self):
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].backward()

    def _batch_update(self):
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].batch_update()

    def _loss(self):
        log_prob = -np.log(
            np.array([self.layers[-1].outputs[j][range(self.max_len), self.labels[j]] for j in range(self.batch_size)]))
        loss = np.sum(log_prob) / self.batch_size
        print("Loss:", loss)

    def train(self, batch_input, batch_label):
        self._forward(batch_input, batch_label)
        self._loss()
        self._backward()
        self._batch_update()

    def predict(self, inputs):
        pass
