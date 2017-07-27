'''
Created on Thursday July
__author__ : 'jdlimingyang@jd.com'
'''
import tensorflow as tf
import numpy as np
print (tf.__version__)

class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        '''
        :param sequence_length: 句子长度
        :param num_classes: 分类的类别数
        :param vocab_size: 词汇的size
        :param embedding_size: 词嵌入的维度
        :param filter_sizes: 卷积核的size
        :param num_filters: 卷积核的个数
        :param l2_reg_lambda: l2正则化系数
        '''

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_normal([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution  + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters], name='b'))
                conv = tf.nn.conv2d(
                    self.embedded_chars_expand,
                    W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='conv'
                )

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length-filter_size+1, 1, 1], # 池化的窗口，只需要在embedding的vector上做max pooling 所以其他维度为1
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='pool'
                )
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs) # 在维度3上合并数据
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add  dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized ) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.drop, W, b, name='scores')
            self.predcitions = tf.argmax(self.scores, 1, name='predicitons')

        # CalculateMean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predcitions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


