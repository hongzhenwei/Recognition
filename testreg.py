import  os
import tensorflow as tf

#1.导入数据
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
data = read_data_sets('MNIST_data', one_hot = True)

#2.创建模型
x = tf.placeholder(tf.float32, [None, 784])
sess=tf.Session()
#选择模型 Y=W*x+b    W[784,10]*x[10*none]+b[1,10]
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
y = tf.nn.softmax(tf.matmul(x, W) + b) #softmax激励函数

#3.模型优化损失函数
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) #交叉熵评估  y_真值  y预测值

#4.训练模型
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#以交叉熵(损失函数)做优化目标,最小化模型评估 基于梯度下降法

#5.测试模型 tf.argmax()获取数据最大值索引 y,y_是one_hot向量
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#6.创建保存训练的模型信息的saver对象
saver = tf.train.Saver([W,b])

#7.开始训练


# sess.run(tf.global_variables_initializer()) #变量初始化
saver.restore(sess, 'mnist/data/regresstion.ckpt')
# print(sess.run(accuracy,feed_dict={x:data.test.images,y_:data.test.labels}))
# sess.run(y,feed_dict={x:})
print(sess.run(y,feed_dict={x:data.test.images[0].reshape(1,784)})) # 标记7
# print(sess.run(y,feed_dict={x:data.test.images[1].reshape(1,784)})) #标记2


#实现测试一个样本判断概率输出
