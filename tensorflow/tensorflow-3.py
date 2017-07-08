
# coding: utf-8

# In[15]:


import tensorflow as tf
import numpy as np
import IPython


# In[2]:


# normal method
a = tf.constant(2,name='a')
b = tf.constant(3,name='b')

x = tf.add(a,b)

with tf.Session() as sess:
    print(sess.run(x))


# In[3]:


# using tensor board

a = tf.constant(2,name='a1')
b = tf.constant(3,name='b1')

x = tf.add(a,b,name='add')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    print(sess.run(x))

writer.close()


# In[4]:


# Constant types

a = tf.constant([2,2],name='vector')
b = tf.constant([[1,2],[3,4]],name='matrix')

with tf.Session() as sess:
    display(sess.run(a))
    display(sess.run(b))


# In[5]:


# tensors whoes elements are specific

tf.zeros([2,3],tf.float32)
input_tensor = tf.ones([5,2])

tf.zeros_like(input_tensor)

tf.ones([2,3],tf.int32)
tf.ones_like(input_tensor)

# fill a tensor with a similar value
tf.ones([3,4],4)


# In[6]:


# create constants that are sqeuences

tf.linspace(10.0,15.0,10,name='linspace')

start = 3
limit = 20
delta = 2

tf.range(start,limit,delta)

tf.range(10,1,-0.5)

tf.range(limit)


# In[7]:


#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
#name=None)
#tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
#name=None)
#tf.random_shuffle(value, seed=None, name=None)
#tf.random_crop(value, size, seed=None, name=None)
#tf.multinomial(logits, num_samples, seed=None, name=None)
#tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)


# In[8]:


# math operations
a = tf.constant([3,2])
b = tf.constant([2,2])
tf.add(a,b)
tf.add_n([a,b,b])   # a + b + b
tf.multiply(a,b)    # element wise
#tf.matmul(a,b)      # error
tf.matmul(tf.reshape(a,shape=[1,2]),tf.reshape(b,shape=[2,1]))
tf.div(a,b)
tf.mod(a,b)


# In[9]:


# data types

t_0 = 20           # treated as 0 d array
tf.zeros_like(t_0) # ===> 0
tf.ones_like(t_0)  # ===> 1

t_1 = ["apple","banana","orange"]          # ==> treated as 1-D array
tf.zeros_like(t_1)                         # ==> ['','','']
#tf.ones_like(t_1)                         # ==> error   

t_2 = [[ True , False , False ],
       [ False , False , True ],
       [ False , True , False ]]
tf.zeros_like(t_2)
tf.ones_like(t_2)


# In[10]:


# numpy data types

tf.ones([2,2],np.float32)


# In[11]:


# building the graphs 
##   10 (2 6) ( 3 )  =  360
##            ( 5 )
#numpy
m1 = np.array([[2.,6.]])
m2 = np.array([[3.],[5.]])

prod = 10 * np.dot(m1,m2)
print('matmul(numpy) = ',prod)

# tensorflow
# build the graph
tm1 = tf.constant([[2.,6.]],name='tm1')
tm2 = tf.constant([[3.],[5.]],name='tm2')
product = 10 * tf.matmul(tm1,tm2,name='product')

# execute the graph
with tf.Session() as sess:
    print('matmul(tesorflow) = ',sess.run(product))


# In[14]:


# using tensorboard
a = tf.constant(4,name='a') 
b = tf.constant(6,name='b')
add = tf.add(a,b,name='add')

with tf.Session() as sess:
    # to activate tensorboard
    writer = tf.summary.FileWriter('./tensorboard',sess.graph)  # create a folder named tensorboard
    
    print('sum = ',sess.run(add))                               # print sum

writer.close()                                                  # close the tensorboard

#Next, go to Terminal, run the program. Make sure that your present working directory is the
#same as where you ran your Python code.
# $ python [ yourprogram . py ]
# $ tensorboard -- logdir = "./graphs"


# In[ ]:




