from modules import *
from data_load import load_vocab

import tensorflow as tf
import tensorflow.contrib.rnn as core_rnn_cell
from tensorflow.contrib import legacy_seq2seq as seq2seq


class TransformerDecoder:
    def __init__(self, is_training=True, args=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.int32, shape=(None, args.maxlen + 1 if args.weight_tying else args.maxlen))

            if is_training:
                self.y = tf.placeholder(tf.int32, shape=(None, args.maxlen + 1 if args.weight_tying else args.maxlen))

            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec, self.lookup_table = embedding(self.x,
                                                        vocab_size=args.vocab_size,
                                                        num_units=args.hidden_units,
                                                        scale=True,
                                                        zero_pad=False,
                                                        scope="dec_embed")
                ## Positional Encoding
                if args.sinusoid:
                    self.dec += positional_encoding(self.x,
                                                    num_units=args.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0),
                                                  [tf.shape(self.x)[0], 1]),
                                          vocab_size=args.maxlen,
                                          num_units=args.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe")[0]

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=args.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(args.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=args.hidden_units,
                                                       num_heads=args.num_heads,
                                                       dropout_rate=args.dropout_rate,
                                                       is_training=is_training,
                                                       causality=args.weight_tying,
                                                       scope="self_attention")

                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=args.hidden_units,
                                                       num_heads=args.num_heads,
                                                       dropout_rate=args.dropout_rate,
                                                       is_training=is_training,
                                                       causality=args.weight_tying,
                                                       scope="vanilla_attention")

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * args.hidden_units, args.hidden_units])
                # Final linear projection
                self.logits = tf.matmul(tf.reshape(self.dec, [-1, args.hidden_units]),
                                        self.lookup_table if args.weight_tying else
                                        tf.get_variable("proj", [args.target_vocab_size, args.hidden_units]),
                                        transpose_b=True)

            # self.logits = tf.layers.dense(self.dec, len(word2idx))
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            if is_training:
                one_hot_depth = args.vocab_size if args.weight_tying else args.target_vocab_size
                # Loss
                self.labels = tf.reshape(
                    tf.stop_gradient(tf.one_hot(self.y, depth=one_hot_depth)), [-1, one_hot_depth])
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
                self.mean_loss = tf.reduce_mean(self.loss)

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter(args.logdir, self.graph)


class RNN:
    def __init__(self, is_training=False, args=None):
        self.args = args
        num_layers = args.num_layers
        weight_tying = args.weight_tying
        infer = not is_training
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = core_rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = core_rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = core_rnn_cell.LSTMCell
        else:
            raise Exception(
                "model type not supported: {}. Make sure that your model is the one of rnn,gru or lstm".format(
                    args.model))
        if num_layers == 1:
            # Init cell with configured number of hidden states
            self.cell = cell_fn(args.rnn_size if not weight_tying else args.embedding_size,
                                reuse=tf.get_variable_scope().reuse)
        else:
            if not weight_tying:
                cells = [cell_fn(args.rnn_size) for _ in range(num_layers)]
            else:
                cells = [cell_fn(args.rnn_size) for _ in range(num_layers - 1)]
                cells.append(cell_fn(args.embedding_size))
            self.cell = core_rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        if args.dropout and (not (infer or args.validation)):
            print("using dropout")
            self.cell = core_rnn_cell.DropoutWrapper(cell=self.cell, output_keep_prob=0.6, seed=1)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name="input_data")
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = self.cell.zero_state(args.batch_size, dtype=tf.float32)

        with tf.variable_scope(args.scope):
            embedding = tf.get_variable("embedding", [args.vocab_size, args.embedding_size])

            # weights
            if not weight_tying:
                softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            else:
                softmax_w = tf.transpose(embedding, name="softmax_w")
            # bias
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), args.seq_length, 1)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

            def loop(prev, i):
                if infer:
                    prev = tf.matmul(prev, softmax_w) + softmax_b
                    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                    return tf.nn.embedding_lookup(embedding, prev_symbol)
                else:
                    return tf.nn.embedding_lookup(embedding, self.input_data[:, i])

            outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, self.cell,
                                                      loop_function=loop if infer else None, scope='decoder')
            # print(last_state.name)
            if not weight_tying:
                output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
            else:
                output = tf.reshape(tf.concat(outputs, 1), [-1, args.embedding_size])
            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits, name="softmax")
            loss = seq2seq.sequence_loss_by_example([self.logits],
                                                    [tf.reshape(self.targets, [-1])],
                                                    [tf.ones([args.batch_size * args.seq_length])],
                                                    args.vocab_size)
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

            self.final_state = last_state
            self.lr = tf.Variable(0.0, trainable=False)
            # print tf.trainable_variables()

            if infer:
                self.values, self.indices = tf.nn.top_k(tf.slice(self.probs, [0, 0], [1, args.vocab_size]), k=10,
                                                        name="top_k")
            else:
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                                  args.grad_clip)
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars))
