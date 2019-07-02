import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets('MNIST_data', one_hot=False)
# print(mnist.train.images[0].shape)  #(784,)
img0 = mnist.test.images[0].reshape(28,28) # 矩阵 二维数组
img1 = mnist.test.images[1].reshape(28,28)
img2 = mnist.test.images[2].reshape(28,28)
img3 = mnist.test.images[3].reshape(28,28)
print(mnist.test.images[0])
# print(type(mnist.test.images[0]))

fig = plt.figure()
# 参数221的意思是：将画布分割成2行2列，图像画在从左到右从上到下的第1块
ax1 = fig.add_subplot(221)
ax1.imshow(img0, cmap=plt.cm.gray)
ax2 = fig.add_subplot(222)
ax2.imshow(img1, cmap=plt.cm.gray)
ax3 = fig.add_subplot(223)
ax3.imshow(img2, cmap=plt.cm.gray)
ax4 = fig.add_subplot(224)
ax4.imshow(img3, cmap=plt.cm.gray)

plt.show()