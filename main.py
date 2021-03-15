from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import pickle
import utils
from model import NVDM

np.random.seed(0)
tf.compat.v1.set_random_seed(0)

flags = tf.app.flags
flags.DEFINE_string('data_dir', './datasets', 'Data dir path.')
flags.DEFINE_string('dataset', 'imdb', 'Data dir path.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_hidden', 500, 'Size of each hidden layer.')
flags.DEFINE_integer('n_topic', 100, 'Size of stochastic vector.')
flags.DEFINE_integer('n_class', 2, 'Size of stochastic vector.')
flags.DEFINE_integer('n_hflow', 0, 'Number of Householder flows.')
flags.DEFINE_integer('n_sample', 1, 'Number of samples.')
flags.DEFINE_integer('vocab_size', 10000, 'Vocabulary size.')
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
flags.DEFINE_integer('embedding_size', 100, 'Dimension of word vector.')
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'tanh', 'Non-linearity of the MLP.')
FLAGS = flags.FLAGS


def train(sess, model,
          train_url,
          test_url,
          n_class,
          batch_size,
          training_epochs=1000,
          alternate_epochs=10):
    """train nvdm model."""
    train_target, train_set, train_count = utils.data_set(train_url)
    test_target, test_set, test_count = utils.data_set(test_url)
    # hold-out development dataset
    dev_target = test_target[:50]
    dev_set = test_set[:50]
    dev_count = test_count[:50]

    dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
    test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)

    mini_ppx = 9999.0
    no_decent_flg = 0

    for epoch in range(training_epochs):
        train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)

        # -------------------------------
        # train
        for switch in range(0, 3):
            if switch == 0:
                optim = model.optim_cls
                print_mode = 'updating classifier'
            elif switch == 1:
                optim = model.optim_dec
                print_mode = 'updating decoder'
            else:
                optim = model.optim_enc
                print_mode = 'updating encoder'
            for i in range(alternate_epochs):
                loss_sum = 0.0
                ppx_sum = 0.0
                recon_sum = 0.0
                kld_sum = 0.0
                cls_sum = 0.0
                word_count = 0
                doc_count = 0
                for idx_batch in train_batches:
                    target_batch, data_batch, count_batch, num_example = utils.fetch_data(
                        train_target, train_set, train_count, idx_batch, n_class, FLAGS.vocab_size)
                    # mask_valid_length = int(0.2 * num_example)
                    # mask = np.array([1.0] * mask_valid_length + [0.0] * (num_example - mask_valid_length))
                    input_feed = {model.x.name: data_batch, model.y.name: target_batch,
                                  model.batch_size.name: num_example}
                    _, (elbo, recon, kld, cls) = sess.run((optim,
                                               [model.elbo, model.recons_loss, model.kld, model.classify_loss]),
                                              input_feed)
                    loss_sum += np.sum(elbo)
                    recon_sum += np.sum(recon) / num_example
                    kld_sum += np.sum(kld) / num_example
                    cls_sum += np.sum(cls) / num_example
                    word_count += np.sum(count_batch)
                    # to avoid nan error
                    # count_batch = np.add(count_batch, 1e-12)
                    # per document loss
                    ppx_sum += np.sum(np.divide(elbo, count_batch))
                    doc_count += num_example
                print_ppx = np.exp(loss_sum / word_count)
                print_ppx_perdoc = np.exp(ppx_sum / doc_count)
                print_recon = recon_sum / len(train_batches)
                print_kld = kld_sum / len(train_batches)
                print_cls = cls_sum / len(train_batches)
                print('| Epoch train: {:d} |'.format(epoch + 1),
                      print_mode, '{:d}'.format(i),
                      '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
                      '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
                      '| RECON: {:.5}'.format(print_recon),
                      '| KLD: {:.5}'.format(print_kld),
                      '| CLS: {:.5}'.format(print_cls))
        # -------------------------------
        # dev
        loss_sum = 0.0
        recon_sum = 0.0
        kld_sum = 0.0
        cls_sum = 0.0
        ppx_sum = 0.0
        word_count = 0
        doc_count = 0
        for idx_batch in dev_batches:
            target_batch, data_batch, count_batch, num_example = utils.fetch_data(
                dev_target, dev_set, dev_count, idx_batch, n_class, FLAGS.vocab_size)
            # mask = np.array([1.0] * num_example)
            input_feed = {model.x.name: data_batch, model.y.name: target_batch, model.batch_size.name: num_example}
            elbo, recon, kld, cls = sess.run([model.elbo, model.recons_loss, model.kld, model.classify_loss],
                                 input_feed)
            loss_sum += np.sum(elbo)
            recon_sum += np.sum(recon) / num_example
            kld_sum += np.sum(kld) / num_example
            cls_sum += np.sum(cls) / num_example
            word_count += np.sum(count_batch)
            # count_batch = np.add(count_batch, 1e-12)
            ppx_sum += np.sum(np.divide(elbo, count_batch))
            doc_count += num_example
        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_recon = recon_sum / len(train_batches)
        print_kld = kld_sum / len(dev_batches)
        print_cls = cls_sum / len(dev_batches)
        print('| Epoch dev: {:d} |'.format(epoch + 1),
              '| Perplexity: {:.9f}'.format(print_ppx),
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
              '| RECON: {:.5}'.format(print_recon),
              '| KLD: {:.5}'.format(print_kld),
              '| CLS: {:.5}'.format(print_cls))
        # -------------------------------
        # test
        if FLAGS.test:
            loss_sum = 0.0
            recon_sum = 0.0
            kld_sum = 0.0
            cls_sum = 0.0
            ppx_sum = 0.0
            word_count = 0
            doc_count = 0
            for idx_batch in test_batches:
                target_batch, data_batch, count_batch, num_example = utils.fetch_data(
                    test_target, test_set, test_count, idx_batch, n_class, FLAGS.vocab_size)
                # mask = np.array([1.0] * num_example)
                input_feed = {model.x.name: data_batch, model.y.name: target_batch, model.batch_size.name: num_example}
                elbo, kld, cls = sess.run([model.elbo, model.kld, model.classify_loss],
                                     input_feed)
                loss_sum += np.sum(elbo)
                recon_sum += np.sum(recon) / num_example
                kld_sum += np.sum(kld) / num_example
                cls_sum += np.sum(cls) / num_example
                word_count += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)
                ppx_sum += np.sum(np.divide(elbo, count_batch))
                doc_count += num_example
            print_ppx = np.exp(loss_sum / word_count)
            print_ppx_perdoc = np.exp(ppx_sum / doc_count)
            print_recon = recon_sum / len(train_batches)
            print_kld = kld_sum / len(test_batches)
            print_cls = cls_sum / len(test_batches)
            print('| Epoch test: {:d} |'.format(epoch + 1),
                  '| Perplexity: {:.9f}'.format(print_ppx),
                  '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
                  '| RECON: {:.5}'.format(print_recon),
                  '| KLD: {:.5}'.format(print_kld),
                  '| CLS: {:.5}'.format(print_cls))

            ckpt_path = "./ckpt/model"
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            # implement of early-stop
            if no_decent_flg < 10:
                if mini_ppx > print_ppx:
                    mini_ppx = print_ppx
                    no_decent_flg = 0
                    path = saver.save(sess, ckpt_path)
                    print("saved model to {}\n".format(path))
                else:
                    no_decent_flg += 1
            else:
                print("optimization finished!\n")
                print("best validating ppx = {:.2f}\n".format(mini_ppx))
                break


def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
        non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
        non_linearity = tf.nn.sigmoid
    else:
        non_linearity = tf.nn.relu

    glove_url = os.path.join(FLAGS.data_dir, 'glove.6B.100d.txt')
    data_dir = os.path.join(FLAGS.data_dir, FLAGS.dataset)
    vocab_url = os.path.join(data_dir, 'vocab.new')
    word_dict = utils.build_vocab(vocab_url)
    glove = utils.load_glove(glove_url, FLAGS.embedding_size, word_dict)

    nvdm = NVDM(vocab_size=FLAGS.vocab_size,
                n_hidden=FLAGS.n_hidden,
                n_topic=FLAGS.n_topic,
                n_class=FLAGS.n_class,
                n_hflow=FLAGS.n_hflow,
                n_sample=FLAGS.n_sample,
                embedding_size=FLAGS.embedding_size,
                learning_rate=FLAGS.learning_rate,
                non_linearity=non_linearity,
                pretrained_embeddings=glove)
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    train_url = os.path.join(data_dir, 'train.feat')
    test_url = os.path.join(data_dir, 'test.feat')
    train(sess, nvdm, train_url, test_url, FLAGS.n_class, FLAGS.batch_size)

    graph = tf.compat.v1.get_default_graph()
    matrix = graph.get_tensor_by_name("decoder/projection/matrix:0")
    pickle.dump(matrix.eval(session=sess), open('matrix.pkl', 'wb'))


if __name__ == '__main__':
    tf.compat.v1.app.run()
