from __future__ import print_function
import argparse
import os

from data_load import next_batch, load_vocab, load_train_data
from modules import *
import pickle
from models import TransformerDecoder
from tone_utils import clear_all_marks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='corpora/blog.txt')
    parser.add_argument('--vocab_path', type=str, default='./ckpt_blog_td/vocab.txt')
    parser.add_argument('--tgt_vocab_path', type=str, default='./ckpt_blog_td/vocab.txt')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt_blog_td2')
    parser.add_argument('--logdir', type=str, default='./ckpt_blog_td2')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--maxlen', type=int, default=30)
    parser.add_argument('--weight_tying', type=int, default=0)
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--sinusoid', type=int, default=0)

    args = parser.parse_args()
    word2idx, idx2word = load_vocab(args.vocab_path)

    if not args.weight_tying:
        ref_ = []
        ref = []
        for idx, token in idx2word.items():
            cleared = clear_all_marks(token)
            if cleared not in ref_:
                ref_.append(cleared)
            ref.append(ref_.index(cleared))

        ref = np.asarray(ref).astype(np.float)
        args.target_vocab_size = len(word2idx)
        args.vocab_size = ref.shape[0]
    else:
        args.vocab_size = len(word2idx)

    with open(os.path.join(args.logdir, "args.pkl"), 'wb') as f:
        pickle.dump(args, f)
    # Construct graph
    model = TransformerDecoder(is_training=True, args=args)
    print("Graph loaded")
    X = load_train_data(args.train_path, args.vocab_path, args.maxlen)
    pad_idx = word2idx["<pad>"]
    num_batch = len(X) // args.batch_size

    # Start session
    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
        if ckpt:
            print("restoring from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        for epoch in range(1, args.num_epochs + 1):
            gs = sess.run(model.global_step)
            for step in range(num_batch - 1):
                if args.weight_tying:
                    x_step, y_step = next_batch(X, step, args.batch_size, args.maxlen, pad_idx)
                else:
                    x_step, _ = next_batch(X, step, args.batch_size, args.maxlen, pad_idx)
                    y_step = x_step[:, 1:]
                    x_step = ref[y_step.astype(np.int)]
                [_, mean_loss, loss] = sess.run([model.train_op, model.mean_loss, model.merged],
                                                feed_dict={
                                                    model.x: x_step,
                                                    model.y: y_step
                                                })
                if step % 10 == 0:
                    model.train_writer.add_summary(loss, gs + step)
                print("epoch = {}, step = {}/{}, loss = {:.4f}".format(epoch, step, num_batch, mean_loss))
            saver.save(sess, args.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")
