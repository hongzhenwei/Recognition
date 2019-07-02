

import tensorflow as tf


x = tf.placeholder("float", [None, 784])
sess = tf.Session()

#获取模型参数
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver([W, b])
saver.restore(sess, 'mnist/data/regresstion.ckpt')

def regression(input):
    result= sess.run(y, feed_dict={x: input}).flatten().tolist()
    return result


