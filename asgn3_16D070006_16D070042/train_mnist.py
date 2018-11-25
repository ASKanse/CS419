from __future__ import print_function

import tensorflow as tf
import math
import numpy as np

from model import myNeuralNet

#def compute_hidden(lay,h_v,p_h):
#	cvh = sess.run(lay, feed_dict={p_h:h_v})
#	return cvh

def compute_accuracy(v_x, v_y):
	global prediction
	#global hidden1
    #input v_x to nn and get the result with y_pre
    #h1_o = compute_hidden(hidden1,v_x,x)   ##################
	y_pre = sess.run(prediction, feed_dict={x : v_x})
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy,feed_dict={x: v_x, y: v_y})
	return result

def get_predictions(v_t):
	y_test = sess.run(prediction, feed_dict={x : v_t})
	return y_test

def add_layer(inputs, in_size, out_size, activation_function=None,regularizer = None):
   
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if regularizer is None:
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b,)
	else:
		if activation_function is None:
			outputs = Wx_plus_b + regularizer(Weights)
		else:
			outputs = activation_function(Wx_plus_b,) + regularizer(Weights)
	return outputs

# x denotes features, y denotes labels
xtrain = np.load('data/mnist/xtrain.npy')
ytrain = np.load('data/mnist/ytrain.npy')

xval = np.load('data/mnist/xval.npy')
yval = np.load('data/mnist/yval.npy')

xtest = np.load('data/mnist/xtest.npy')

dim_input = 784
dim_output = 10

learn_rate = 0.5
batch_size = 3000 #100
epoch = 5 #5


train_size = len(xtrain)
valid_size = len(xval)
test_size = len(xtest)
 
spe = math.floor(train_size/batch_size)
total_steps = spe*epoch

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#h1 = tf.placeholder(tf.float32, [None, 10])
#add layers
#hidden1 = add_layer(x, 784, 10, activation_function=tf.nn.relu)
	#hidden2 = add_layer(hidden1, 500, 250, activation_function=tf.nn.relu)
	#hidden3 = add_layer(hidden2, 250, 100, activation_function=tf.nn.relu)
prediction = add_layer(x, 784 ,10, activation_function=tf.nn.softmax, regularizer=None)
#calculate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
#use Gradientdescentoptimizer
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
m = 0
for i in range(epoch):
	for j in range(spe):
		batch_x, batch_y = xtrain[m:m+batch_size],ytrain[m:m+batch_size]
		sess.run(train_step,feed_dict={x: batch_x, y: batch_y})
		if i*j % 100 == 0:
			print(compute_accuracy(xval, yval))
	m += 100
test_pred = get_predictions(xtest)
for i in test_pred:
	print(i)
