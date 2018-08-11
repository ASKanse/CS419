# load the csv file 
# use th data to build the tree model 
# pass the tree to sinfer.py
# choose the best split recursively
# maximise the var_gain
# for continous variable sort and then find split condition
# for each node calc var_gain by deciding on the split condition

import csv
train_dat = open('train.csv',newline='')
readfile = csv.reader(train_dat)
attributes = next(readfile)

print(attributes)
data = []
for row in readfile:
	lstatr = []
	for j in range(len(attributes)):
		lstatr.append(float(row[j]))
	data.append(lstatr)
data.sort()
#for dat in data:
#	print(dat)
#print(type(data))

allattr_list = []
for j in range(len(attributes)):
	attri = []
	for i in range(len(data)):
		vals = []
		vals.append(data[i][j])
		vals.append(data[i][11])
		attri.append(vals)
	attri.sort()
	allattr_list.append(attri)

#################################################################

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

###################################################################

def var_split_point(attrlst):
	sp = 0
	gain = 0
	prt = []
	for i in range(len(attrlst)):
		prt.append(attrlst[i][1]) 
	for i in range(len(attrlst)):
		lst1 = []
		lst2 = []
		p = attrlst[i][0]
		for j in range(len(attrlst)):
			if attrlst[j][0] < p:
				lst1.append(attrlst[j][1])
			else:
				lst2.append(attrlst[j][1])
		if len(lst1) == 0:
			lst1 = [0]
		elif len(lst2) == 0:
			lst2 =[0]
		if var_gain(prt,lst1,lst2) > gain:
			gain =  var_gain(prt,lst1,lst2)
			sp = p
	return [sp,gain]

###################################################################

def calc_abserr(lst_dat):
	summ = 0
	for i in range(len(lst_dat)):
		summ = summ + lst_dat[i]
	mean = (summ/len(lst_dat))
	abserr = 0
	for i in range(len(lst_dat)):
		val = lst_dat[i]
		abserr = abserr + (mean-val)
	return abserr

def abs_gain(pl,cl1,cl2):
	return calc_abserr(pl) - (((len(cl1)/len(pl))*calc_abserr(cl1))+((len(cl2)/len(pl))*calc_abserr(cl2)))

#####################################################################

def abs_split_point(attrlst):
	sp = 0
	gain = 0
	prt = []
	for i in range(len(attrlst)):
		prt.append(attrlst[i][1]) 
	for i in range(len(attrlst)):
		lst1 = []
		lst2 = []
		p = attrlst[i][0]
		for j in range(len(attrlst)):
			if attrlst[j][0] < p:
				lst1.append(attrlst[j][1])
			else:
				lst2.append(attrlst[j][1])
		if len(lst1) == 0:
			lst1 = [0]
		elif len(lst2) == 0:
			lst2 =[0]
		if abs_gain(prt,lst1,lst2) > gain:
			gain =  abs_gain(prt,lst1,lst2)
			sp = p
	return [sp,gain]

#print(var_split_point(attr1))
#print(abs_split_point(attr1))
#########################################################################
def splitlist(nd,l,sp):
	lst1 = []
	lst2 = []
	for i in range(len(l)):
		if l[i] < sp:
			lst1.append(l[i])
		else:
			lst2.append(l[i])
	nd.rchildlst = lst1
	nd.lchildlst = lst2

#########################################################################
#class node -- self variables = attribute , list to split , right and left child attribute name

class node:

	def __init__(self):
		self.attrname = ''
		self.lst = []
		self.rchildlst = []
		self.lchildlst = []
# any functions needed in the class???

#recurrsively find the attribute with the maximum variance gain
attribute_list = allattr_list

maxindex = 0
for i in range(len(attribute_list)):
	max_gain = 0
	var_sp_gain = var_split_point(attribute_list[i])
	if var_sp_gain[1] > max_gain:
		max_gain = var_sp_gain[1]
		maxindex = i

attr = node()
attr.attrname = attributes[maxindex]
for i in range(len(attribute_list[maxindex])):
	attr.lst.append(attribute_list[maxindex][i][1])
spl = var_split_point(attribute_list[maxindex])
splitlist(attr,attr.lst,spl[0])

print(attr.attrname)



