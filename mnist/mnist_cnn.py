# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#训练集的Image
x = tf.placeholder(tf.float32, [None, 784]) #X(?,784)
#训练集的Label
y_actual = tf.placeholder(tf.float32, [None, 10]) #y(?,10)

# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #加入噪音 避免0梯度
    return tf.Variable(initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #参数1 绿色 参数2 黄色 参数3 步长1 参数4 卷积核处理方式

# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #参数1 卷积层输出x 参数2 2*2模板 各缩小1/2 参数3同2


x_image = tf.reshape(x, [-1, 28, 28, 1]) #初始化图片数据格式  -1 忽略 28*28 特征集 1 灰度通道

# 构建网络
W_conv1 = weight_variable([5, 5, 1, 32]) #5*5卷积核  输入通道1 算出32个特征 输出通道32
b_conv1 = bias_variable([32]) #每个通道有一个对应偏置

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层 提取特征 非线性化
h_pool1 = max_pool(h_conv1)  # 第一个池化层  将原特征按提取特征压缩

W_conv2 = weight_variable([5, 5, 32, 64])#5*5卷积核 输入通道32,输出通道64
b_conv2 = bias_variable([64])  #每个通道有一个对应偏置
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层 提取特征 非线性化
h_pool2 = max_pool(h_conv2)  # 第二个池化层 将现有特征按提取特征压缩 1/2

W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 2次池化缩小1/4 7*7 输出通道1024
b_fc1 = bias_variable([1024])#每个通道有一个对应偏置
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 全连接层  y=x*W+b


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout层 优化 减少过拟合 一个神经元输出保持概率不变

W_fc2 = weight_variable([1024, 10])#第二层全连接  输入通道1024 输出通道10
b_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层
        # y=softmax(x*w+b)

cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))  # 交叉熵
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #更复杂的梯度下降

#模型评价
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 50 == 0:  # 训练1000次，验证一次
            train_acc = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 1.0})
            print('step', i, 'training accuracy', train_acc)

        train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

    test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
    print("test accuracy", test_acc)
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'convalutional.ckpt')
    )

    print('Save:', path)

