import tensorflow as tf
import numpy as np
import math
# include any other imports that you want

'''
This file contains a class for you to implement your neural net.
Basic function skeleton is given, and some comments to guide you are also there.

You will find it convenient to look at the tensorflow API to understand what functions to use.
'''

'''
Implement the respective functions in this class
You might also make separate classes for separate tasks , or for separate kinds of networks (normal feed-forward / CNNs)
'''
class myNeuralNet:
	# you can add/modify arguments of *ALL* functions
	# you might also add new functions, but *NOT* remove these ones
	def __init__(self, dim_input_data, dim_output_data,xtrain,ytrain): # you can add/modify arguments of this function 
		# Using such 'self-ization', you can access these members in later functions of the class
		# You can do such 'self-ization' on tensors also, there is no change
		self.dim_input_data = dim_input_data
		self.dim_output_data = dim_output_data
		self.xtrain= xtrain
		self.ytrain= ytrain

		# Create placeholders for input : data as well as labels
		self.ip = tf.placeholder(tf.float32, shape = (None,dim_input_data))
		self.op = tf.placeholder(tf.float32, shape = (None,dim_output_data))
		# You might want to initialising some container to store all the layers of the network
		self.layer_container = []
	def addHiddenLayer(self, layer_dim_in, layer_dim_out, layer_number, activation_fn=None, regularizer_fn=None):
		# Add a layer to the network of layer_dim
		ipl = []
		if (self.layer_container == []):
			ipl = self.ip
		else :
			ipl = self.layer_container[layer_number-1]
		Weights = tf.Variable(tf.random_normal([layer_dim_in, layer_dim_out]))
		#init b: a matric in 1*y
		biases = tf.Variable(tf.zeros([1, layer_dim_out]) + 0.01,)
		#calculate the result
		Wx_plus_b = tf.matmul(ipl, Weights) + biases
		#add the active hanshu
		if activation_fn is None:
			hidden = Wx_plus_b
		else:
			hidden = activation_fn(Wx_plus_b,)
		'''
		ipl = []
		if (self.layer_container == []):
			ipl = self.ip
		else :
			ipl = self.layer_container[layer_number-1]
		hidden = tf.layers.dense(inputs=ipl , units=layer_dim, activation=activation_fn, regularizer = regularizer_fn)'''                         
		# It might be a good idea to append the new layer to the container of layers that you initialized before
		self.layer_container.append(hidden)

	def addFinalLayer(self, activation_fn=None, regularizer_fn=None):
		# We don't take layer_dim here, since the dimensionality of final layer is
		# already stored in self.dim_output_data
		# Create the output of the final layer as logits
		output = tf.layers.dense(inputs=self.layer_container[-1], units=self.dim_output_data, activation= tf.nn.softmax)
		self.logits = output 
		self.layer_container.append(output)
		# You might also like to apply the final activation function (softmax / sigmoid) to get the predicted labels
	
	def setup_training(self, learn_rate):
		# Define loss, you might want to store it as self.loss
		self.loss = tf.losses.softmax_cross_entropy(self.ytrain, self.logits)
		# Define the train step as self.train_step = ..., use an optimizer from tf.train and call minimize(self.loss)
		train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)

	def setup_metrics(self):
		# Use the predicted labels and compare them with the input labels(placeholder defined in __init__)
		count = 0
		for i in range(self.dim_input_data):
			if (self.op[i] == self.logits[i]):
				count += 1
		self.accuracy = count/self.dim_input_data 
		# to calculate accuracy, and store it as self.accuracy
		pass
	
	# you will need to add other arguments to this function as given below
	def train(self, sess, max_epochs, batch_size, train_size, print_step = 100): # valid_size, test_size, etc
		# Write your training part here
		# sess is a tensorflow session, used to run the computation graph
		# note that all the functions uptil now were just constructing the computation graph
		
		# one 'epoch' represents that the network has seen the entire dataset once - it is just standard terminology
		steps_per_epoch = int(train_size/batch_size)
		max_steps = max_epochs * steps_per_epoch
		for step in range(max_steps):
			# read a batch of data from the training data

			# now run the train_step, self.loss on this batch of training data. something like :
			_, train_loss = sess.run([self.train_step, self.loss], feed_dict={'''here, feed in your placeholders with the data you read in the previous to previous comment'''})
			if (step % print_step) == 0:
				# read the validation dataset and report loss, accuracy on it by running
				val_acc, val_loss = sess.run([self.accuracy, self.loss], feed_dict={'''here, feed in your placeholders with the data you read in the comment above'''})
				# remember that the above will give you val_acc, val_loss as numpy values and not tensors
				pass
			# store these train_loss and validation_loss in lists/arrays, write code to plot them vs steps
			# Above curves are *REALLY* important, they give deep insights on what's going on
		# -- for loop ends --
		# Now once training is done, run predictions on the test set
		test_predictions = sess.run('''here, put something like self.predictions that you would have made somewhere''', feed_dict={'''here, feed in test dataset'''})
		return test_predictions
		# This is because we will ask you to submit test_predictions, and some marks will be based on how your net performs on these unseen instances (test set)
		'''
		We have done everything in train(), but
		you might want to create another function named eval(),
		which calculates the predictions on test instances ...
		'''

	'''
	NOTE:
	you might find it convenient to make 3 different train functions corresponding to the three different tasks,
	and call the relevant one from each train_*.py
	The reason for this is that the arguments to the train() are different across the tasks
	'''
	'''
	Example, for the speech part, the train() would look something like :
	(NOTE: this is only a rough structure, we don't claim that this is exactly what you have to do.)
	
	train(self, sess, batch_size, train_size, max_epochs, train_signal, train_lbls, valid_signal, valid_lbls, test_signal):
		steps_per_epoch = math.ceil(train_size/batch_size)
		max_steps = max_epochs*steps_per_epoch
		print(max_steps)
		for step in range(max_steps):
			# select batch_size elements randomly from training data
			sampled_indices = random.sample(range(train_size), batch_size)
			trn_signal = train_signal[sampled_indices]
			trn_labels = train_lbls[sampled_indices]
			if (step % steps_per_epoch) == 0:
				val_loss, val_acc = sess.run([self.loss, self.accuracy], feed_dict={input_data: valid_signal, input_labels: valid_lbls})
				print(step, val_acc, val_loss)
			sess.run(self.train_step, feed_dict={input_data: trn_signal, input_labels: trn_labels})
		test_prediction = sess.run([self.predictions], feed_dict={input_data: test_signal})
		return test_prediction
	'''