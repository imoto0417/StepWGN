#
# coding: utf-8

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# trainデータの件数
print(mnist.train.num_examples)
# testデータの件数
print(mnist.test.num_examples)
# validationデータの件数
print(mnist.validation.num_examples)
