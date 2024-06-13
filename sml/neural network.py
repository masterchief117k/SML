import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
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
left =2.5
top = 2.5
fig = plt.figure(figsize=(10,10))
for i in range(6):
    ax=fig.add.subplot(3,2,i+1)
    im=np.reshape(mnist.train.images[i,:],[28,28])
    label=np.argmax(mnist.train.label[i,:])
    ax.imshow(im,cmap='Greys')
    ax.text(left,top,str(label))
plt.show()
inputs = tf.placeholder(tf.float32,[None,784])
Weights = tf.Variable(tf.zeros([784,10]))
bias =tf.Variable(tf.zeros([10]))
outputs = tf.nn.softmax(tf.matmul(inputs , Weights) + bias)
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.reduce_sum(y_*tf.log(outputs), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(number_of_steps):
    batch_xs,batch_ys=mnist.train.next_batch(batch_size)
    sess.run(train_step,feed_dict={inputs: batch_xs,y_:batch_ys})
correct_prediction=tf.equal(tf.argmax(outputs,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction ,tf.float32))
print("Accuracy: ",sess.run(accuracy,feed_dict={inputs:mnist.test.images,y_:mnist.test.labels}))
