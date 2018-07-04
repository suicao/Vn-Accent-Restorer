from __future__ import print_function

import argparse
import pickle
import tensorflow as tf
import numpy as np
from data_load import load_vocab, basic_tokenizer
from models import TransformerDecoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prime', type=str, default='chỉ cần em')
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default="./logdir2_blog")
    parser.add_argument('--vocab_path', type=str, default="./corpora/blog_vocab.txt")
    parser.add_argument('--saved_args_path', type=str, default="./logdir2_blog/args.pkl")

    args = parser.parse_args()
    with open(args.saved_args_path, 'rb') as f:
        saved_args = pickle.load(f)

    g = TransformerDecoder(is_training=False, args=saved_args)

    word2idx, idx2word = load_vocab(args.vocab_path)
    unk_idx = word2idx.get("<unk>")
    pad_idx = word2idx.get("<pad>")
    sos_idx = word2idx.get("<sos>")
    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))

        prime = np.asarray([sos_idx] + [word2idx.get(w, unk_idx) for w in basic_tokenizer(args.prime)])
        prime_len = prime.shape[0]
        prime = np.atleast_2d(
            np.lib.pad(prime, [0, saved_args.maxlen + 1 - len(prime)], 'constant', constant_values=pad_idx))
        softmax = tf.nn.softmax(g.logits)
        probs = sess.run(softmax, feed_dict={g.x: prime})[prime_len - 1]
        print([idx2word[w] for w in np.argsort(-probs)[:10]])
        result = []
        for i in range(args.seq_length):
            pred = sess.run(g.preds, feed_dict={g.x: prime})[prime_len - 1]
            result.append(idx2word[pred])
            if prime_len >= prime.shape[1]:
                prime[0][:-1] = prime[0][1:]
            else:
                prime_len += 1
            prime[0][prime_len - 1] = pred
        print(' '.join(result))
