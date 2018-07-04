# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf

import argparse
import os
import _pickle as cPickle
from models import TransformerDecoder, RNN
import tone_utils
from data_load import basic_tokenizer, load_vocab
from tone_utils import clear_all_marks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='ckpt_blog_td/vocab.txt',
                        help='the location of checkpointing files')
    parser.add_argument('--ckpt_path', type=str, default='ckpt_blog_td2',
                        help='the location of checkpointing files')
    parser.add_argument('--model', type=str, default='t-d-s2s',
                        help='model types: t-d, t-d-s2s or rnn')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='beam search size')
    parser.add_argument('--prime', type=str,
                        default='chia tay me di buc ca minh')
    parser.add_argument('--saved_args_path', type=str, default="./ckpt_blog_td2/args.pkl")

    args = parser.parse_args()
    with open(args.saved_args_path, 'rb') as f:
        saved_args = cPickle.load(f)
    word2idx, idx2word = load_vocab(args.vocab_path)
    accent_vocab = dict()
    for token, idx in word2idx.items():
        raw_token = tone_utils.clear_all_marks(token) if token[0] != "<" else token
        if raw_token not in accent_vocab:
            accent_vocab[raw_token] = [token]
        else:
            curr = accent_vocab[raw_token]
            if token not in curr:
                curr.append(token)

    unk_idx = word2idx["<unk>"]
    if args.model == "t-d" or args.model == "t-d-s2s":
        # quick fix
        model = TransformerDecoder(False, saved_args)
    else:
        model = RNN(False, saved_args)

    with tf.Session(graph=model.graph if args.model != "rnn" else None) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            words = basic_tokenizer(args.prime)
            if args.model == "t-d":
                sos_idx = word2idx["<sos>"]
                pad_idx = word2idx["<pad>"]
                init_state = np.full(shape=(saved_args.maxlen + 1), fill_value=pad_idx)
                init_state[0] = sos_idx
                init_probs = sess.run(tf.nn.softmax(model.logits), feed_dict={
                    model.x: np.atleast_2d(init_state)})[0]
                paths = beamsearch_transformer(sess, model,
                                               words, args.beam_size,
                                               saved_args.maxlen, init_probs,
                                               accent_vocab, word2idx)
            elif args.model == "rnn":
                x = np.zeros((1, 1))
                words = basic_tokenizer(args.prime)
                init_state = sess.run(model.cell.zero_state(1, tf.float32))
                if words[0] != "<eos>":
                    words = ["<eos>"] + words
                out_state = init_state
                x[0, 0] = word2idx[words[0]] if words[0] in word2idx else unk_idx
                # print(x[0,0])
                feed = {model.input_data: x, model.initial_state: out_state}
                [probs, out_state] = sess.run([model.probs, model.final_state], feed)
                paths = beamsearch_rnn(sess, model,
                                       words, args.beam_size,
                                       out_state, probs[0],
                                       accent_vocab, word2idx)
            else:
                pad_idx = word2idx["<pad>"]
                ref = []
                for idx, token in idx2word.items():
                    cleared = clear_all_marks(token)
                    if cleared not in ref:
                        ref.append(cleared)
                words = basic_tokenizer(args.prime)
                feed_x = np.asarray([ref.index(w) for w in words])
                feed_x = np.atleast_2d(
                    np.lib.pad(feed_x, [0, saved_args.maxlen - len(feed_x)], 'constant', constant_values=pad_idx))
                feed = {model.x: feed_x}
                paths = [sess.run(model.preds, feed_dict=feed)]
                paths[0][len(words):] = pad_idx
            result = ""
            for path in paths:
                for idx, token in enumerate(path):
                    result += idx2word[token] if token != unk_idx else words[idx if args.model != "rnn" else idx + 1]
                    result += " "
                result += "\n"
            print(result)


def find_candidates(probs, nums_suggestion, word, vocab, accent_vocab):
    if word not in accent_vocab:
        return np.array([vocab["<unk>"]]), np.array([1])
    else:
        possible_candidates = np.asarray([vocab[x] for x in accent_vocab[word]])
        possible_candidates_probs = probs[possible_candidates]
        top_k = np.argsort(-possible_candidates_probs)[:nums_suggestion]
        possible_candidates = possible_candidates[top_k]
        return possible_candidates, probs[possible_candidates]


def beamsearch_transformer(sess, model, sent, nums_suggestion,
                           maxlen, initial_probs,
                           accent_vocab, vocab):
    # top_k = np.argsort(-initial_probs[0])[:nums_suggestion]
    stop_token = vocab["<pad>"]
    num_tokens = len(sent)
    state_probs = np.zeros(nums_suggestion, dtype=np.float32)
    all_candidates = np.zeros([nums_suggestion, nums_suggestion], dtype=np.int32)
    all_candidates_probs = np.zeros([nums_suggestion, nums_suggestion], dtype=np.float32)
    paths = np.zeros([nums_suggestion, num_tokens], dtype=np.int32)
    states = np.zeros([nums_suggestion, maxlen + 1], dtype=np.int32)
    sum_probs = 0.0

    # initializing first graph level
    top_k, top_k_probs = find_candidates(initial_probs, nums_suggestion, sent[0], vocab, accent_vocab)
    for i in range(nums_suggestion):
        state_probs[i] = initial_probs[top_k[i]] if i < len(top_k) else 0
        sum_probs += state_probs[i]
        paths[i, 0] = top_k[i] if i < len(top_k) else stop_token

    for i in range(nums_suggestion):
        state_probs[i] /= sum_probs
        for level in range(1, num_tokens):
            paths[i, level] = stop_token
        states[i, 1:num_tokens + 1] = paths[i].copy()
        states[i, 0] = vocab["<sos>"]
    top_k = list(top_k)
    # for each graph level
    for level in range(1, num_tokens):
        for s_idx in range(0, nums_suggestion):
            if s_idx >= len(top_k):
                top_k.append(top_k[0])
            feed = states[s_idx].copy()
            initial_probs = sess.run(model.logits, feed_dict={model.x: np.atleast_2d(feed)})[level]
            initial_probs = sess.run(tf.nn.softmax(initial_probs))
            # top_prob_indices = np.argsort(-initial_probs[0])[:nums_suggestion]
            top_prob_indices, _ = find_candidates(initial_probs, nums_suggestion, sent[level], vocab, accent_vocab)

            for k in range(len(top_prob_indices)):
                sum_probs += state_probs[s_idx] * initial_probs[top_prob_indices[k]]
            for k in range(nums_suggestion):
                all_candidates[s_idx, k] = top_prob_indices[k] if k < len(top_prob_indices) else -1
                all_candidates_probs[s_idx, k] = state_probs[s_idx] * initial_probs[
                    top_prob_indices[k]] / sum_probs if k < len(top_prob_indices) else 0
        # find top k among all_candidates_probs
        top_candidates = np.argsort(-all_candidates_probs.reshape(-1))[:nums_suggestion]
        tmp_path = np.zeros([nums_suggestion, num_tokens], dtype=np.int32)
        for s_idx in range(0, nums_suggestion):
            top_k[s_idx] = all_candidates.flat[top_candidates[s_idx]]
            state_probs[s_idx] = all_candidates_probs.flat[top_candidates[s_idx]]
            state_idx = int(top_candidates[s_idx] / nums_suggestion)
            # update path
            for c in range(num_tokens):
                if c < level:
                    tmp_path[s_idx, c] = paths[state_idx, c]
                else:
                    tmp_path[s_idx, c] = stop_token
            tmp_path[s_idx, level] = top_k[s_idx]
            states[s_idx, 1:num_tokens + 1] = tmp_path[s_idx]

            sum_probs += state_probs[s_idx]

        for s_idx in range(nums_suggestion):
            state_probs[s_idx] /= sum_probs
        paths = tmp_path
    return paths


def beamsearch_rnn(sess, model, sent, nums_suggestion,
                   initial_state, initial_probs,
                   accent_vocab, vocab):
    # top_k = np.argsort(-initial_probs[0])[:nums_suggestion]
    stop_token = vocab["<eos>"]
    states = []
    nums_token = len(sent) - 1
    state_probs = np.zeros(nums_suggestion, dtype=np.float32)
    all_candidates = np.zeros([nums_suggestion, nums_suggestion], dtype=np.int32)
    all_candidates_probs = np.zeros([nums_suggestion, nums_suggestion], dtype=np.float32)
    paths = np.zeros([nums_suggestion, nums_token], dtype=np.int32)
    sum_probs = 0.0

    # initializing first graph level
    top_k, top_k_probs = find_candidates(initial_probs, nums_suggestion, sent[1], vocab, accent_vocab)
    for i in range(nums_suggestion):
        states.append(initial_state)
        state_probs[i] = initial_probs[top_k[i]] if i < len(top_k) else 0
        sum_probs += state_probs[i]
        paths[i, 0] = top_k[i] if i < len(top_k) else -1
    for i in range(nums_suggestion):
        state_probs[i] /= sum_probs
        for level in range(1, nums_token):
            paths[i, level] = stop_token
    top_k = list(top_k)
    # for each graph level
    for level in range(1, nums_token):
        values = []
        for s_idx in range(0, nums_suggestion):
            if s_idx >= len(top_k):
                top_k.append(top_k[0])
            token = top_k[s_idx]
            x = np.zeros((1, 1))
            x[0, 0] = token
            feed = {model.input_data: x, model.initial_state: states[s_idx]}
            [initial_probs, state_] = sess.run([model.probs, model.final_state], feed)
            initial_probs = initial_probs[0]
            values.append((initial_probs, state_))

            # top_prob_indices = np.argsort(-initial_probs[0])[:nums_suggestion]
            top_prob_indices, _ = find_candidates(initial_probs, nums_suggestion, sent[level + 1], vocab, accent_vocab)

            for k in range(len(top_prob_indices)):
                sum_probs += state_probs[s_idx] * initial_probs[top_prob_indices[k]]
            for k in range(nums_suggestion):
                all_candidates[s_idx, k] = top_prob_indices[k] if k < len(top_prob_indices) else -1
                all_candidates_probs[s_idx, k] = state_probs[s_idx] * initial_probs[
                    top_prob_indices[k]] / sum_probs if k < len(top_prob_indices) else 0
        # find top k among all_candidates_probs
        top_candidates = np.argsort(-all_candidates_probs.reshape(-1))[:nums_suggestion]
        tmp_path = np.zeros([nums_suggestion, nums_token], dtype=np.int32)
        for s_idx in range(0, nums_suggestion):
            top_k[s_idx] = all_candidates.flat[top_candidates[s_idx]]
            state_probs[s_idx] = all_candidates_probs.flat[top_candidates[s_idx]]
            state_idx = int(top_candidates[s_idx] / nums_suggestion)
            states[s_idx] = values[state_idx][1]
            # update path
            for c in range(nums_token):
                if c < level:
                    tmp_path[s_idx, c] = paths[state_idx, c]
                else:
                    tmp_path[s_idx, c] = -1
            tmp_path[s_idx, level] = top_k[s_idx]

            sum_probs += state_probs[s_idx]

        for s_idx in range(nums_suggestion):
            state_probs[s_idx] /= sum_probs
        paths = tmp_path
    return paths


if __name__ == '__main__':
    main()
