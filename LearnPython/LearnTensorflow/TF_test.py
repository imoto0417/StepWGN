# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import time
import tensorflow as tf
import sys
print(sys.version)
#import numpy as np
#import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# trainデータの件数
print(mnist.train.num_examples)
# testデータの件数
print(mnist.test.num_examples)
# validationデータの件数
print(mnist.validation.num_examples)
