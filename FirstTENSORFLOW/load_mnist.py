# -*- coding: utf-8 -*-
import warnings
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

warnings.filterwarnings('ignore')


trainX, trainY, testX, testY, = mnist.load_data('./data/mnist/', one_hot=True)

# print(len(trainX), len(trainY))

# print(trainX)

# print(trainY)

# print(trainX[0], trainY[0])

plt.imshow(trainX[1].reshape(28, 28), cmap=cm.gray_r, interpolation='nearest')
plt.show()
