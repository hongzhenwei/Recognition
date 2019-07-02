import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
data = read_data_sets('MNIST_data', one_hot = True)
#卷积操作
x = tf.placeholder("float", [None, 784])
keep_prob = tf.placeholder("float")

sess = tf.Session()

# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 构建网络
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层
h_pool1 = max_pool(h_conv1)  # 第一个池化层

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层
h_pool2 = max_pool(h_conv2)  # 第二个池化层

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 全连接层

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout层

W_fc2 = weight_variable([1024, 10])#第二层全连接
b_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层

saver = tf.train.Saver()
saver.restore(sess, 'mnist/data/convalutional.ckpt')


 #标记2
print(sess.run(y_predict, feed_dict={x: data.test.images[0].reshape(1,784), keep_prob: 1.0}))

sess.close()

#实现测试一个样本判断概率输出
