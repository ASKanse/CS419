import numpy as np
import math

def square_hinge_loss(targets, outputs):
	l = len(outputs)
	sum1 = 0.0
	for i in range(l):
		if((targets[i])*outputs[i] <= 1):
			sum1+= (1-(targets[i])*outputs[i])**2
	return sum1/l

def logistic_loss(targets, outputs):
	# Write thee logistic loss loss here
	l = len(outputs)
	sum1 = 0.0
	for i in range(l):
		sum1+= math.log(1+math.exp(-(targets[i])*outputs[i]))
	return sum1/l

def perceptron_loss(targets, outputs):
	# Write thee perceptron loss here
	l = len(outputs)
	sum1 = 0.0
	for i in range(l):
		if((targets[i])*outputs[i] < 0):
			sum1 -= (targets[i])*outputs[i]
	return sum1/l

def L2_regulariser(weights):
    # Write the L2 loss here
	l = len(weights)-1 #don't include bias term
	sum1 = 0
	for i in range(l):
		sum1+=weights[i]**2
	#return math.sqrt(sum1)
	return sum1

def L4_regulariser(weights):
    # Write the L4 loss here
	l = len(weights)-1 #don't include bias term
	sum1 = 0
	for i in range(l):
		sum1+=weights[i]**4
	#return math.sqrt(math.sqrt(sum1))
	return sum1


def square_hinge_grad(weights,inputs, targets, outputs):
  # Write thee square hinge loss gradient here
	l1 = len(weights)
	result = []
	#results = np.random.random(l1)
	l = len(targets)
	for j in range(l1):
		sum1 = 0
		for i in range(l):
			if((targets[i])*outputs[i] <= 1):
				sum1+= -2*(targets[i])*inputs[i][j]*(1-(targets[i])*outputs[i])
		result.append(sum1)
	return result

def logistic_grad(weights,inputs, targets, outputs):

	# Write thee logistic loss loss gradient here
	l1 = len(weights)
	l = len(targets)
	results = []
	for j in range(l1):
		sum1 = 0
		for i in range(l):
			sum1 += (math.exp(- (targets[i]) * outputs[i])/(1 + math.exp(- (targets[i]) * outputs[i])))*(-inputs[i][j]*(targets[i]))
		results.append(sum1)
	return results

def perceptron_grad(weights,inputs, targets, outputs):
	  # Write thee perceptron loss gradient here
	l1 = len(weights)
	result = []
	l = len(targets)
	for j in range(l1):
		sum1 = 0
		for i in range(l):
			if((targets[i])*outputs[i] < 0):
				sum1+= -1*(targets[i])*inputs[i][j]
		result.append(sum1)
	return result

def L2_grad(weights):
     # Write the L2 loss gradient here
	l = len(weights)-1
	results = []
	#sum1 = 0
	#for i in range(l):
	#	sum1 += weights[i]**2
	for i in range(l):
		#results.append(weights[i]/sum1**0.5)
		results.append(2*weights[i])
	results.append(0)
	return results

def L4_grad(weights):
    # Write the L4 loss gradient here
	l = len(weights)-1
	#sum1 = 0
	results = []
	#for i in range(l):
	#	sum1 += weights[i]**4
	for i in range(l):
		#results.append((weights[i]*weights[i]*weights[i])/(sum1**(0.75)))
		results.append(4*(weights[i]*weights[i]*weights[i]))
	results.append(0)
	return results

loss_functions = {"square_hinge_loss" : square_hinge_loss, 
                  "logistic_loss" : logistic_loss,
                  "perceptron_loss" : perceptron_loss}

loss_grad_functions = {"square_hinge_loss" : square_hinge_grad, 
                       "logistic_loss" : logistic_grad,
                       "perceptron_loss" : perceptron_grad}

regularizer_functions = {"L2": L2_regulariser,
                         "L4": L4_regulariser}

regularizer_grad_functions = {"L2" : L2_grad,
                              "L4" : L4_grad}
