import numpy as np
from model import transformer
from loaddata import load_train_data, load_de_vocab, load_en_vocab


class Train(object):
    def __init__(self):
        self.batch_size = 32
        self.max_len = 10
        self.min_cnt = 20

        self.epoch = 10000

        self.model = None
        self.initialize_network()

    def initialize_network(self):
        de2idx, idx2de = load_de_vocab(self.min_cnt)
        en2idx, idx2en = load_en_vocab(self.min_cnt)
        source_word_count = len(en2idx)
        target_word_count = len(de2idx)
        self.model = transformer.Transformer(self.batch_size, source_word_count, target_word_count, self.max_len)

    def train(self):
        train_x, train_y = load_train_data(self.min_cnt, self.max_len)
        print("Done load all data.")
        for epoch in range(self.epoch):
            select = np.random.randint(0, train_x.shape[0], self.batch_size)
            train_x_batch, train_y_batch = train_x[select, :], train_y[select, :]

            self.model.train(train_x_batch, train_y_batch)


if __name__ == '__main__':
    tr = Train()
    tr.train()
