#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'admin'
__time__ = '2019/12/30 2:53 下午'

import tensorflow as tf
import random
import numpy as np


def data_set(data_url):
    """process data input."""
    data = []
    target = []
    word_count = []
    fin = open(data_url)
    while True:
        line = fin.readline()
        if not line:
            break
        id_freqs = line.split()
        doc = {}
        count = 0

        for id_freq in id_freqs[1:]:
            items = id_freq.split(':')
            # python starts from 0
            doc[int(items[0])-1] = int(items[1])
            count += int(items[1])
        if count > 0:
            target.append(int(id_freqs[0]))
            data.append(doc)
            word_count.append(count)
    fin.close()
    return target, data, word_count


def create_batches(data_size, batch_size, shuffle=True):
    """create index by batches."""
    batches = []
    ids = list(range(data_size))
    if shuffle:
        random.shuffle(ids)
    # 最后的 minibatch 需要补零
    num_batches = np.ceil(data_size / batch_size).astype('int')
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, data_size)
        batches.append(ids[start:end])
    return batches


def fetch_data(target, data, count, idx_batch, n_class, vocab_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    target_batch = np.zeros((batch_size, n_class))
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    indices = []
    values = []
    for i, doc_id in enumerate(idx_batch):
        for word_id, freq in data[doc_id].items():
            data_batch[i, word_id] = freq
        target_batch[i, target[doc_id]] = 1.0
        count_batch.append(count[doc_id])
    return target_batch, data_batch, count_batch, batch_size


def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix."""
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
    return ret_list


'''
#===================================================================
# 需要固定明确指出 batch_size
def get_householder_matrix(vec, dims, batch_size):
    vec_list = tf.split(vec, batch_size, axis=0)
    householder_matrix_list = []
    for v in vec_list:
        vec_mod = tf.scalar_mul(tf.constant(2.0), tf.math.reciprocal(tf.add(tf.reduce_sum(tf.square(v)), 1.0))) # 2 * (1 / || v ||^2 )
        householder_matrix_list.append(tf.linalg.tensor_diag(tf.ones([dims])) - tf.scalar_mul(vec_mod, tf.matmul(v, v, transpose_a=True))) # Householder Matrix
    return tf.stack(householder_matrix_list) # [ batch_size, dims, dims] Tensor


def householder_transform(vec, mat, dims, batch_size):
    vec_list = tf.split(vec, batch_size, axis=0)
    mat_list = tf.split(mat, batch_size, axis=0)
    result_list = []
    for i in range(0, batch_size):
        result_list.append(tf.matmul(vec_list[i], tf.squeeze(mat_list[i])))
    return tf.concat(result_list, axis=0)  # [ batch_size, dims] Tensor
#===================================================================
'''


def householder_transform(z, v, dims):
    """
    计算householder flow
    :param z: Tensor, [None, dims]
    :param v: Tensor, [None, dims]
    :return: Tensor, [None, dims]
    """
    A = tf.matmul(tf.expand_dims(v, 2), tf.expand_dims(v, 1)) # Tensor, [None, dims, dims]
    Az = tf.reshape(tf.matmul(A, tf.expand_dims(z, -1)), (-1, dims)) # Tensor, [None, dims]
    v_norm_sq = tf.reduce_sum(tf.square(v), axis=1, keepdims=True) # Tensor, [None, 1]
    return z - 2 * Az / v_norm_sq


def load_glove(glove_file, embedding_size, word_dict):
    print('loading glove pre-trained word embeddings ...')
    embedding_weights = {}
    f = open(glove_file, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_weights[word] = vector
    f.close()
    print('total {} word vectors in {}'.format(len(embedding_weights), glove_file))

    embedding_matrix = np.random.uniform(-0.5, 0.5, (len(word_dict), embedding_size)) / embedding_size

    oov_count = 0
    for i, word in word_dict.items():
        embedding_vector = embedding_weights.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov_count += 1
    print('number of OOV words = %d' % oov_count)
    return embedding_matrix


def build_vocab(vocab_url):
    with open(vocab_url) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    word_dict = {}
    for i, line in enumerate(lines):
        word_dict[i] = line[0]
    return word_dict