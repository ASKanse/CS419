
# coding: utf-8

# In[176]:


import pandas as pd
import numpy as np


# In[177]:


data = pd.read_csv('train.csv')
data2 = pd.read_csv('test.csv')
data.head()


# In[178]:


def rss_of_partition(data_column,output_column):
    step_size=(data_column.max()-data_column.min())/150
    minimum_rss=((output_column.mean()-output_column)**2).sum()  
    current_pointer=data_column.min()
    final_pointer=data_column.min()
    for i in range(150):
        current_pointer=current_pointer+step_size*(i)
        data_on_left=output_column[data_column<current_pointer]
        data_on_right=output_column[data_column>=current_pointer]
        mean_on_left=data_on_left.mean()       
        mean_on_right=data_on_right.mean()
                
        if ((mean_on_left-output_column[data_column<current_pointer])**2).sum()+((mean_on_right-output_column[data_column>=current_pointer])**2).sum()<minimum_rss:
            minimum_rss=((mean_on_left-output_column[data_column<current_pointer])**2).sum()+((mean_on_right-output_column[data_column>=current_pointer])**2).sum()
            final_pointer=current_pointer

    return minimum_rss,final_pointer


# In[179]:


def best_attribute_split(data_matrix):
    columns_names=list(data_matrix.columns.values)[0:data_matrix.columns.size-1]
    minimum_rss,split_for_minimum_rss=rss_of_partition(data_matrix[columns_names[0]],data_matrix[data_matrix.columns[data_matrix.columns.size-1]])
    atributename=columns_names[0]
    
    for names in columns_names:
        rss,split=rss_of_partition(data_matrix[names],data_matrix[data_matrix.columns[data_matrix.columns.size-1]])
        if rss<minimum_rss:
            minimum_rss=rss
            split_for_minimum_rss=split
            atributename=names
    return split_for_minimum_rss,atributename            


# In[180]:


class node:
    def __init__(self,data=None):
        self.split_value=None
        self.split_attribute=None
        self.left_child=None
        self.right_child=None
        self.data=data
        self.rss=None
        self.mean=None
        self.ldata=None
        self.rdata=None
        self.abs=None
        self.parent=None


# In[181]:


def tree(root,min_size,loss):
    root.mean=((root.data)[data.columns[data.columns.size-1]]).mean()
    root.split_value,root.split_attribute=best_attribute_split(root.data)
    root.ldata=(root.data[root.data[root.split_attribute]<root.split_value])
    root.rdata=(root.data[root.data[root.split_attribute]>=root.split_value])
    root.abs=(abs(root.mean-root.data[data.columns[data.columns.size-1]])).sum()
    root.rss=((root.mean-root.data[data.columns[data.columns.size-1]])**2).sum()        
    if (root.ldata).shape[0]>=min_size and (root.rdata).shape[0]>=min_size:
        root.left_child=node(root.ldata)
        root.right_child=node(root.rdata)
        loss = loss + tree(root.left_child,min_size,loss)
        loss = loss + tree(root.right_child,min_size,loss)
    else:
        loss=loss+(root.rss)
    return loss


# In[182]:


root=node(data)
rss_error=tree(root,15,0)


# In[183]:


rss_error


# In[184]:


def give_output(root,data_row):
    if root.left_child != None:
        if (data_row[root.split_attribute]<root.split_value):
            return give_output(root.left_child,data_row)
        else:
            return give_output(root.right_child,data_row)
    
    return root.mean
        
    
def create_output_data(data_matrix):
    index = range(0,data_matrix.shape[0])
    dataoutput = pd.DataFrame(index=index,columns=[['Id',data.columns[data.columns.size-1]]])
    i=0
    while (i < data_matrix.shape[0]):
        dataoutput.iloc[i]=i+1,give_output(root,data_matrix.iloc[i])
        i=i+1
    return dataoutput   


# In[185]:


output=create_output_data(data2)
output.to_csv('output.csv',index=False)


# In[ ]:




