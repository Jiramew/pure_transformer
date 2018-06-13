import numpy as np
import codecs
import regex


def load_de_vocab(min_cnt):
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab(min_cnt):
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(min_cnt, maxlen, source_sents, target_sents):
    en2idx, idx2en = load_en_vocab(min_cnt)
    de2idx, idx2de = load_de_vocab(min_cnt)

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        y = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
        x = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <= maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), maxlen], np.int32)
    Y = np.zeros([len(y_list), maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_train_data(min_cnt, maxlen):
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                codecs.open('corpora/train.tags.de-en.de', 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                codecs.open('corpora/train.tags.de-en.en', 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]

    X, Y, Sources, Targets = create_data(min_cnt, maxlen, de_sents, en_sents)
    return X, Y


def load_test_data(min_cnt, maxlen):
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

    de_sents = [_refine(line) for line in
                codecs.open('corpora/IWSLT16.TED.tst2014.de-en.de.xml', 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in
                codecs.open('corpora/IWSLT16.TED.tst2014.de-en.en.xml', 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]

    X, Y, Sources, Targets = create_data(min_cnt, maxlen, de_sents, en_sents)
    return X, Sources, Targets  # (1064, 150)
