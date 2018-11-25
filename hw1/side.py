from sys import argv
side.py, train.csv, test.csv, 



import csv
from operator import itemgetter

train_dat = open('train2.csv',newline='')
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

def splitpoint(dat):
	i = 0
	maxgain = 0
	spf = 0
	indx = 0

	lst = []
	for m in range(len(dat)):
		last = dat[m][len(dat[i])-1]
		lst.append(last)

	for j in range(len(dat[i])-1):
		
		sorted(dat, key = itemgetter(j))
		for l in range(len(dat)):
			sp = dat[l][j]
			lst1 = []
			lst2 = []
			for k in range(len(dat)):
				if dat[k][j] < sp:
					lst1.append(dat[k][len(dat[i])-1])
				else:
					lst2.append(dat[k][len(dat[i])-1])
			if len(lst1) == 0:
				lst1 = [0]
			if len(lst2) == 0:
				lst2 = [0]
			g = var_gain(lst, lst1, lst2)
			if(g > maxgain):
				maxgain =  g
				spf = sp
				indx = j
			else:
				continue
	return [indx , spf , maxgain]

print(splitpoint(data))

def splitdata(data , split_point): #should return two data sets based on the split_point(indx and spf)
	left, right = list(), list()
	for row in data:
		if row[split_point[0]] < split_point[1]:
			left.append(row)
		else:
			right.append(row)
	return [left, right]

class node:
	def __init__(self):
		self.attrname = ''
		self. = []
		self.sp = 0
		self.indx = 0
		self.rcdata = []
		self.lcdata = []

tree = []

def form_tree(dat): # recurssive function for left and right half of the tree
	splitinfo = splitpoint(dat)
	attr = node()
	[attr.rcdata,attr.lcdata] = splitdata(dat , [splitinfo[0],splitinfo[1]])
	form_tree(attr.rcdata)
	form_tree(attr.lcdata)
