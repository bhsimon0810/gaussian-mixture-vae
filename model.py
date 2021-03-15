import numpy as np
import tensorflow as tf
import utils
from modules import linear, mlp


class NVDM(object):
    """ Neural Variational Document Model -- BOW VAE.
    """

    def __init__(self,
                 vocab_size,
                 n_hidden,
                 n_topic,
                 n_class,
                 n_hflow,
                 n_sample,
                 learning_rate,
                 non_linearity,
                 embedding_size,
                 pretrained_embeddings):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_class = n_class
        self.n_hflow = n_hflow
        self.n_sample = n_sample
        self.embedding_size = embedding_size
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.pretrained_embeddings = pretrained_embeddings

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.vocab_size], name='input')
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.n_class], name='output')
        self.batch_size = tf.compat.v1.placeholder(tf.int32, name="batch_size")
        # self.padding_mask = tf.compat.v1.placeholder(tf.float32, [None], name='padding_mask')
        # self.mask = tf.compat.v1.placeholder(tf.float32, [None], name='label_mask')

        # classifier
        with tf.compat.v1.variable_scope('classifier'):
            self.embeddings = tf.compat.v1.get_variable(
                name="embeddings",
                shape=[self.vocab_size, self.embedding_size],
                initializer=tf.constant_initializer(self.pretrained_embeddings),
                dtype=tf.float32
            )
            self.scores = mlp(tf.matmul(self.x, self.embeddings), [self.n_hidden, self.n_class], self.non_linearity)
            self.y_hat = tf.nn.softmax(self.scores)
            self.classify_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.y)


        # encoder
        with tf.compat.v1.variable_scope('encoder'):
            self.enc_vec = mlp(self.x, [self.n_hidden], self.non_linearity)
            self.mean = {}
            self.logsigm = {}
            self.klds = {}

            for i in range(self.n_class):
                self.mean[i] = linear(self.enc_vec, self.n_topic, scope='mean' + str(i))
                self.logsigm[i] = linear(self.enc_vec,
                                      self.n_topic,
                                      bias_start_zero=True,
                                      matrix_start_zero=True,
                                      scope='logsigm' + str(i))

                self.klds[i] = -0.5 * tf.reduce_sum(1 - tf.square(self.mean[i]) + 2 * self.logsigm[i] - tf.exp(2 * self.logsigm[i]), 1)

            kld1 = tf.math.log(tf.cast(self.n_class, tf.float32)) + tf.reduce_sum(
                tf.multiply(self.y_hat, tf.math.log(self.y_hat)), axis=1)
            gmm = tf.stack([kld for idx, kld in self.klds.items()], axis=1)
            kld2 = tf.reduce_sum(tf.multiply(gmm, self.y_hat), axis=1)
            self.kld = kld1 + kld2

        # decoder
        with tf.compat.v1.variable_scope('decoder'):
            self.z = {}
            self.v = {}
            self.v[1] = linear(self.enc_vec, self.n_topic, scope='v1')
            if self.n_sample == 1:  # single sample
                eps = tf.random.normal((self.batch_size, self.n_topic), 0, 1)
                doc_vec = {}
                for i in range(self.n_class):
                    doc_vec[i] = tf.multiply(tf.exp(self.logsigm[i]), eps) + self.mean[i]

                self.z[0] = tf.squeeze(
                    tf.matmul(tf.stack([vec for idx, vec in doc_vec.items()], axis=2), tf.expand_dims(self.y_hat, -1)),
                    axis=2)

                # househodler flow
                for i in range(1, self.n_hflow + 1):
                    self.z[i] = utils.householder_transform(self.z[i-1], self.v[i], self.n_topic)
                    if i == self.n_hflow:
                        break
                    self.v[i + 1] = linear(self.v[i], self.n_topic, scope='v' + str(i + 1))

                logits = tf.nn.log_softmax(linear(self.z[self.n_hflow], self.vocab_size, scope='projection'))
                self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)


            # multiple samples
            else:
                eps = tf.random.normal((self.n_sample * self.batch_size, self.n_topic), 0, 1)
                eps_list = tf.split(eps, self.n_sample, axis=0)
                recons_loss_list = []
                for i in range(self.n_sample):
                    if i > 0: tf.compat.v1.get_variable_scope().reuse_variables()
                    curr_eps = eps_list[i]
                    doc_vec = {}
                    for i in range(self.n_class):
                        doc_vec[i] = tf.multiply(tf.exp(self.logsigm[i]), curr_eps) + self.mean[i]
                    self.z[0] = tf.squeeze(tf.matmul(tf.stack([vec for idx, vec in doc_vec.items()], axis=2),
                                                     tf.expand_dims(self.y_hat, -1)), axis=2)
                    # househodler flow
                    for i in range(1, self.n_hflow + 1):
                        self.z[i] = utils.householder_transform(self.z[i - 1], self.v[i], self.n_topic)
                        if i == self.n_hflow:
                            break
                        self.v[i + 1] = linear(self.v[i], self.n_topic, scope='v' + str(i + 1))
                    logits = tf.nn.log_softmax(linear(self.z[self.n_hflow], self.vocab_size, scope='projection'))
                    recons_loss_list.append(-tf.reduce_sum(tf.multiply(logits, self.x), 1))
                self.recons_loss = tf.add_n(recons_loss_list) / self.n_sample

        self.elbo = self.recons_loss + self.kld
        self.objective = self.elbo + self.classify_loss

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        fullvars = tf.compat.v1.trainable_variables()

        cls_vars = utils.variable_parser(fullvars, 'classifier')
        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')

        cls_grads = tf.gradients(self.objective, cls_vars)
        enc_grads = tf.gradients(self.objective, enc_vars)
        dec_grads = tf.gradients(self.objective, dec_vars)

        self.optim_cls = optimizer.apply_gradients(zip(cls_grads, cls_vars))
        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))
