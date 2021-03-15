#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'admin'
__time__ = '2019/12/30 2:54 下午'

import tensorflow as tf


def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None):
    with tf.compat.v1.variable_scope(scope or 'Linear'):
        if matrix_start_zero:
            matrix_initializer = tf.constant_initializer(0)
        else:
            matrix_initializer = None
        if bias_start_zero:
            bias_initializer = tf.constant_initializer(0)
        else:
            bias_initializer = None
        input_size = inputs.get_shape()[1].value
        matrix = tf.compat.v1.get_variable('matrix', [input_size, output_size],
                                           initializer=matrix_initializer)
        bias_term = tf.compat.v1.get_variable('bias', [output_size],
                                              initializer=bias_initializer)
        output = tf.matmul(inputs, matrix)
        if not no_bias:
            output = output + bias_term
    return output


def mlp(inputs,
        mlp_hidden=[],
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):

    """Define an MLP."""
    with tf.compat.v1.variable_scope(scope or 'Linear'):
        mlp_layer = len(mlp_hidden)
        res = inputs
        for l in range(mlp_layer):
            res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)))
    return res


'''
def cnn(inputs,
        num_classes,
        num_filters,
        embedding_size,
        sequence_length,
        dropout_keep_prob=0.5,
        scope=None,
        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        pooled_outputs = []

        # kernel_size = 3, 4, 5
        kernel3_shape = [3, embedding_size, 1, num_filters]
        kernel3 = tf.get_variable('kernel3', kernel3_shape, initializer=tf.truncated_normal_initializer())
        bias3 = tf.get_variable('bias3', [num_filters])
        conv3 = tf.nn.conv2d(inputs, kernel3, strides=[1, 1, 1, 1], padding="VALID", name="conv3")

        kernel4_shape = [4, embedding_size, 1, num_filters]
        kernel4 = tf.get_variable('kernel4', kernel4_shape, initializer=tf.truncated_normal_initializer())
        bias4 = tf.get_variable('bias4', [num_filters])
        conv4 = tf.nn.conv2d(inputs, kernel4, strides=[1, 1, 1, 1], padding="VALID", name="conv4")

        kernel5_shape = [5, embedding_size, 1, num_filters]
        kernel5 = tf.get_variable('kernel5', kernel5_shape, initializer=tf.truncated_normal_initializer())
        bias5 = tf.get_variable('bias5', [num_filters])
        conv5 = tf.nn.conv2d(inputs, kernel5, strides=[1, 1, 1, 1], padding="VALID", name="conv5")

        # Apply nonlinearity
        h3 = tf.nn.relu(tf.nn.bias_add(conv3, bias3), name="relu3")
        h4 = tf.nn.relu(tf.nn.bias_add(conv4, bias4), name="relu4")
        h5 = tf.nn.relu(tf.nn.bias_add(conv5, bias5), name="relu5")
        # Maxpooling over the outputs
        pooled3 = tf.nn.max_pool(h3, ksize=[1, sequence_length - 3 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool3")
        pooled4 = tf.nn.max_pool(h4, ksize=[1, sequence_length - 4 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool4")
        pooled5 = tf.nn.max_pool(h5, ksize=[1, sequence_length - 5 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool5")

        pooled_outputs.append(pooled3)
        pooled_outputs.append(pooled4)
        pooled_outputs.append(pooled5)

        # Combine all the pooled features
        num_filters_total = num_filters * 3
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

        # Final logits
        W = tf.get_variable(
            "fc_weights",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bias")
        logits = tf.nn.xw_plus_b(h_drop, W, b, name="logits")

    return logits
'''
