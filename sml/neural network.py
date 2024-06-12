import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST/", one_hot=True)
#Define the various Deep Learning parameters as well as the image and label size
#image_size = 28 refers to the pixel dimensions and num_labels = 10 refers to the number of #digits (0-9)
image_size = 28
num_labels = 10
learning_rate = 0.05
number_of_steps = 1000
batch_size = 100
# Define placeholders
x_train = tf.placeholder(tf.float32, [None,image_size*image_size])
y_train = tf.placeholder(tf.float32, [None, num_labels])
print(x_train)
