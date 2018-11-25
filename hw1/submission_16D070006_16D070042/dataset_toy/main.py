import csv
import numpy as np
import time
import argparse

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the variance gain index for a split dataset
def calc_variance(lst_dat):
	summ = 0
	for i in range(len(lst_dat)):
		summ = summ + lst_dat[i]
	mean = (summ/len(lst_dat))
	var = 0
	for i in range(len(lst_dat)):
		val = lst_dat[i]
		var = var + (mean-val)**2
	return var

def var_gain(pl,cl1,cl2):
	return calc_variance(pl) - (((len(cl1)/len(pl))*calc_variance(cl1))+((len(cl2)/len(pl))*calc_variance(cl2)))

#Calculate the variance gain index for a split dataset
def calc_abserr(lst_dat):
	summ = 0
	for i in range(len(lst_dat)):
		summ = summ + lst_dat[i]
	mean = (summ/len(lst_dat))
	abserr = 0
	for i in range(len(lst_dat)):
		val = lst_dat[i]
		abserr = abserr + np.abs(mean-val)
	return abserr

def abs_gain(pl,cl1,cl2):
	return calc_abserr(pl) - (((len(cl1)/len(pl))*calc_abserr(cl1))+((len(cl2)/len(pl))*calc_abserr(cl2)))

# Select the best split point for a dataset by variance
def get_var_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_maxgain, b_groups = 0, 0, 0, None
	parent_list = []
	for row in dataset:
		parent_list.append(row[-1])
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			left, right = groups
			lchild_list, rchild_list = [], []
			for row1 in left:
				lchild_list.append(row[-1])
			for row1 in right:
				rchild_list.append(row[-1])
			if (len(left) == 0 or len (right) == 0):
				continue
			gain = var_gain(parent_list, lchild_list, rchild_list)
			if gain > b_maxgain:
				b_index, b_value, b_maxgain, b_groups = index, row[index], gain, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Select the best split point for a dataset by abs error
def get_abs_split(dataset):
	
	b_index, b_value, b_maxgain, b_groups = 0, 0, 0, None #([[0]],[[0]])
	parent_list = []
	for row in dataset:
		parent_list.append(row[-1])
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			left, right = groups
			lchild_list, rchild_list = [], []
			for row1 in left:
				lchild_list.append(row[-1])
			for row1 in right:
				rchild_list.append(row[-1])
			if (len(lchild_list) == 0 or len (rchild_list) == 0):
				continue
			gain = abs_gain(parent_list, lchild_list, rchild_list)
			if gain > b_maxgain:
				b_index, b_value, b_maxgain, b_groups = index, row[index], gain, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}	
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	#return max(set(outcomes), key=outcomes.count)
	return sum(outcomes)/len(outcomes) #average value
 
# Create child splits for a node or make terminal by varience
def var_split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_var_split(left)
		var_split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_var_split(right)
		var_split(node['right'], max_depth, min_size, depth+1)

# Create child splits for a node or make terminal by abs error
def abs_split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_abs_split(left)
		abs_split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_abs_split(right)
		abs_split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree by variance
def build_var_tree(train, max_depth, min_size):
	root = get_var_split(train)
	var_split(root, max_depth, min_size, 1)
	return root

# Build a decision tree by abs error
def build_abs_tree(train, max_depth, min_size):
	root = get_abs_split(train)
	abs_split(root, max_depth, min_size, 1)
	return root
'''
# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
'''
# to count the number of nodes
def nodecount(node, l = []):
	if isinstance(node, dict):
		l.append(0)
		nodecount(node['left'], l)
		nodecount(node['right'], l)
	else:
		l.append(0)

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
# pruning
#def prune(node, val_data):


# commad line input
parser = argparse.ArgumentParser(description = 'Command line input')
parser.add_argument('--train_data' , help = 'train datset', required = True, type = str)
parser.add_argument('--test_data' , help = 'test datset' , required = True, type = str)
parser.add_argument('--min_leaf_size' , help = 'min num of data points in leaf' , type = int, default = 1)
parser.add_argument('--absolute' , help = 'abs error' , action = 'store_true')
parser.add_argument('--mean_squared' , help = 'mse' , action = 'store_true')
args = parser.parse_args()

# load datasets
train_dat = open(args.train_data,newline='')
readfile = csv.reader(train_dat)
attributes = next(readfile)

print(attributes)

data = []
train_data = []
val_data = []

for row in readfile:
	lstatr = []
	for j in range(len(attributes)):
		lstatr.append(float(row[j]))
	data.append(lstatr)
train_data_length = np.floor(len(data)*(2/3))
filled = 0
for row in data:
	if filled <= train_data_length:
		train_data.append(row)
	else:
		val_data.append(row)
	filled += 1

test_dat = open(args.test_data,newline='')
readfile = csv.reader(test_dat)
attribute = next(readfile)

print(attribute)

test_data = []
for row in readfile:
	lstatr = []
	for j in range(len(attribute)):
		lstatr.append(float(row[j]))
	test_data.append(lstatr)

# train and predict the output
if (args.absolute == True):
	starttime = time.time()
	tree = build_abs_tree(train_data, 12, args.min_leaf_size)
	traintime = float(time.time() - starttime)
	print(traintime)

	predict_matrix = []
	i = 1
	for row in test_data:
		inst = []
		prediction = predict(tree, row)
		inst.append(i)
		inst.append(prediction)
		predict_matrix.append(inst)
		i = i+1
		print('Got=%d' % (prediction))
	predict_output = open('predictions2.csv' , 'w')
	head = ['ID', 'output']
	writer = csv.writer(predict_output)
	writer.writerow(head)
	for row in predict_matrix:
		writer.writerow(row)
	endtime = float(time.time() - starttime)
	print(endtime)

if (args.mean_squared == True):
	starttime = time.time()
	tree = build_var_tree(train_data, 12, args.min_leaf_size)
	traintime = float(time.time() - starttime)
	print(traintime)

	predict_matrix = []
	i = 1
	for row in test_data:
		inst = []
		prediction = predict(tree, row)
		inst.append(i)
		inst.append(prediction)
		predict_matrix.append(inst)
		i = i+1
		print('Got=%d' % (prediction))
	predict_output = open('predictions2.csv' , 'w')
	head = ['ID', 'output']
	writer = csv.writer(predict_output)
	writer.writerow(head)
	for row in predict_matrix:
		writer.writerow(row)
	endtime = float(time.time() - starttime)
	print(endtime)

'''
err = []
for row in err_mat:
	err.append(row[0]-row[1])
mse = (calc_variance(err)/len(err))
l = []
nodecount(tree,l)
print(len(l))
print(mse)

predict_matrix = []
i = 1
for row in test_data:
	inst = []
	prediction = predict(tree, row)
	inst.append(i)
	inst.append(prediction)
	predict_matrix.append(inst)
	i = i+1
	print('Got=%d' % (prediction))

predict_output = open('predictions2.csv' , 'w')
head = ['ID', 'output']
writer = csv.writer(predict_output)
writer.writerow(head)
for row in predict_matrix:
	writer.writerow(row)
'''
#print_tree(tree)