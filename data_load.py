from __future__ import print_function
import numpy as np
import codecs
import regex as re
from tqdm import tqdm

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")
STOP_WORDS = "\" \' [ ] . , ! : ; ?".split(" ")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
        # return [w.lower() for w in words if w not in stop_words and w != '' and w != ' ']
    return [w.lower() for w in words if w != '' and w != ' ']


def load_vocab(path):
    vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_train_data(train_path, vocab_path, maxlen):
    sents = [basic_tokenizer(line) for line in
             codecs.open(train_path, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    word2idx, idx2word = load_vocab(vocab_path)
    unk_idx = word2idx["<unk>"]
    pad_idx = word2idx["<pad>"]
    sos_idx = word2idx["<sos>"]
    # Index
    result = np.zeros(shape=[len(sents), maxlen + 2])
    for idx, sent in tqdm(enumerate(sents)):
        sent = np.asarray([word2idx.get(w, unk_idx) for w in sent[:maxlen + 1]])
        result[idx][0] = sos_idx
        result[idx][1:] = np.lib.pad(sent, [0, maxlen + 1 - len(sent)], 'constant', constant_values=pad_idx)
    return result


def next_batch(X, step, batch_size, maxlen, pad_idx):
    x = X[step * batch_size:(step + 1) * batch_size, :maxlen + 1].copy()
    for i in range(x.shape[0]):
        first_pad = np.where(x[i] == pad_idx)[0]
        if len(first_pad) > 0 and first_pad[0] > 0:
            x[i][first_pad[0] - 1] = pad_idx
    Y = X[step * batch_size:(step + 1) * batch_size, 1:maxlen + 2]
    return x, Y
