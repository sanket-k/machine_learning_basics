
# coding: utf-8

# In[2]:


#importing lib
import numpy as np
import time 
import sys


# In[5]:


# Memory usage
# python list
py_list = range(1000) #python list
print('python list takes',sys.getsizeof(7)*len(py_list),'bytes') # 7* can be any number

#numpy array
np_array = np.arange(1000)
print('numpy array takes',np_array.size*np_array.itemsize,'bytes')


# In[7]:


# Time taken by numpy and python list
array_size = 10000000 # 10 million ,a consideribly large size helps understand the time difference better

# Python list
py_list1 = range(array_size)
py_list2 = range(array_size)

py_start = time.time() # capturing a time instance(similar to a stop watch)
result = [(x + y) for x,y in zip(py_list1,py_list2)] # performing addiiton of the two lists
py_end = time.time() # capturing a second  time instance(the difference gives the time taken for the operation)
print('time taken for python list :',(py_end - py_start) * 1000,'ms') # By default the time is in seconds *1000 coverts it to milliseonds

# numpy array
np_array1 = np.arange(array_size)
np_array2 = np.arange(array_size)

np_start = time.time()
result = np_array1 + np_array2
np_end = time.time()
print('time taken for numpy array :',(np_end - np_start) * 1000,'ms')

