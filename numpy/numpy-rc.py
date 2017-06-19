
# coding: utf-8

# In[1]:


#importing lib
import numpy as np
import time 
import sys


# In[2]:


# Memory usage
# python list
py_list = range(1000) #python list
print('python list takes',sys.getsizeof(7)*len(py_list),'bytes') # 7* can be any number

#numpy array
np_array = np.arange(1000)
print('numpy array takes',np_array.size*np_array.itemsize,'bytes')
np_array.itemsize,np_array.size


# In[3]:


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


# In[4]:


# Creating arrays
array_a = np.array([1,2,3])   # create 1D array
array_b = np.array([[2,4,5],[5,3,9]],dtype=float)   # creating a 2x3 matrix with values as type float
array_c = np.array([[(3.2,5,6),(4.1,7,9)],[(2,4,5),(7,4,9)]],dtype=float)   # creating multple arrays with data type as float

print(array_a,'= array a (1D array)\n')
print(array_b,'= array b (2D array)\n')
print(array_c,'= array c (multiple arrays)')


# In[5]:


# Creating arrays with place holders

zeros_array = np.zeros((3,3))  # creates a 3x3 matrix with zeros
ones_array = np.ones((3,4))    # creates a 3x4 array with ones
empty_array = np.empty((3,4))  # creates an empty array with dimensions 3x4
eye_array = np.eye(3)          # creates an identity matrix of 3x3
random_array = np.random.random((3,2)) # creates a 3x3 array with random variables
full_array = np.full((2,2),6)  #creates an array with constant number 6

print(zeros_array,'= zeros array \n')
print(ones_array,'= ones array\n')
print(empty_array,'= empty array\n')
print(eye_array,'= identity array\n')
print(random_array,'= random array \n')
print(full_array,'= constant array \n')


# In[6]:


# Array inspection

ins_array = np.array([[3,2,1,4],[7,5,9,2]])   # creating a 2x4 array(inspection array)

print(ins_array,'= Array')  #displaying the array
print('\n',ins_array.shape,'= shape the array / dimensions of the array\n')
print(len(ins_array),'length of the array\n')
print(ins_array.ndim,'= number of array dimensions\n')
print(ins_array.size,'= number of array elements\n')
print(ins_array.dtype,'= data type of the array elements\n')
print(ins_array.dtype.name,'= name of the data type\n')
print(ins_array.astype(float),'= convert an array to a different type\n') 


# In[7]:


# Arithmetic opreations

array_a = np.array([[2,3],[5,7]])  # creating array A 2x2
array_b = np.array([[4,2],[6,8]])  # creating array B 2x2

print(array_a,' = array A\n')
print(array_b,' = array B\n')

print(array_a - array_b,' = subtraction\n')  #subtracting two arrays normal operation
print(np.subtract(array_a,array_b),'= subtraction using numpy\n')

print(array_a + array_b,' = addition\n')  #addition two arrays normal operation
print(np.add(array_a,array_b),'= addition using numpy\n')

print(array_a / array_b,' = division\n')  #dividing two arrays normal operation
print(np.divide(array_a,array_b),'= division using numpy\n')

print(array_a * array_b,' = multiplication\n')  #multiplying two arrays normal operation
print(np.multiply(array_a,array_b),'= multiplication using numpy\n')


# In[8]:


print(array_a,'= array A\n')

print(np.exp(array_a),' = exponentitation\n')   # returns the exponent of the array
print(np.sqrt(array_a),'= Square root\n')       # returns the square root of the array
print(np.sin(array_a),' = sines\n')             # prints the sines of the array
print(np.cos(array_a),' = cosine\n')            # returns element wise cosine
print(np.log(array_a),' = natural logarithm\n') # returns element wise natural logarithm
print(array_a.dot(2),' = dot product\n')        # dot product of the array


# In[9]:


# Comparision

print(array_a,' = array A\n')
print(array_b,' = array B\n')

print(array_a == array_b,' = comparison\n')  # element wise comparison
print(array_a<5 ,' = comparison\n')          # prints true for all values < 5
print('array A and array B are equal = ',np.array_equal(array_a,array_b))  # array wise comparison


# In[10]:


# Aggregate functions

print(array_a,' = array A\n')                   # display array
print(array_a.sum(),'--- sum\n')                # array wise sum 
print(array_a.min(),'--- minimum value\n')      # array wise minimum value
print(array_a.max(axis=0),'--- max value\n')    # maximum value of an array row
print(array_a.mean(),'--- mean\n')              # mean of the array
print(array_a.std(),'--- std\n')                # standard deviation of the array    


# In[11]:


# sorting arrays

rand_array =np.random.random((8,3))                                      # Creating a 8x1 array of random variables

print(rand_array,' = random variable array\n')                           # displaying array
print(np.sort(rand_array,axis=0),' = sorting of array colom wise')      # By default the sorting is applied on row wise axis = 1 changes it to colom wise   


# In[12]:


# Subsetting

array_a = np.random.random(9)                         # create an array of 9 elements
print(array_a,' = array A\n')                         # Display array
print('the 7th element is = ',array_a[6])             # selecting the 7th element in array A

array_b = np.random.randint(5, size=(2,5))            # create random array 
print(array_b,' = array B\n')                         # Display array
print('2nd coloum 2nd row = ',array_b[1,1])           # selecting the element from 2nd row 2nd coloumn


# In[13]:


# slicing

temp_array = np.random.randint(10,size=(2,3))                # Create a random array and store it in a temp array 
#array_a = temp_array                                        # assign array A to the temp array
print(array_a,' = array A\n')                                # display array A
print('the first 4 elements are = ',array_a[0:4],'\n')       # Display till element 4
print(array_a[::-1],' = reversed array A\n')                 # reverse the array A  

#array_b = temp_array                                        # assign array B to the temp array with dimensions 2x3
print(array_b,' = array B\n')                                # Display array B

print(array_b[0:2,1],'= row 0 & 1 in coloumn 1 \n')          # Select items at rows 0 and 1 in coloumn 1
print(array_b[:1],' = row 0 \n')                             # Select all items at row 0


# In[14]:


# boolean indexing

print(array_b,' = array B\n')                                 # Disaplay array B
print(array_b[array_b<3],'\n')                                # Display all elements in array B less than 3


# In[15]:


# array manipulation

# Transposing
print(array_b,' = array B\n')                                 # Dispplay array B
print(np.transpose(array_b),' = transposed array B(numpy)\n') # numpy function for transposing an array
print(array_b.T,' = transposed array B\n')                    # transposing array 

# Changing shape of the array
print(array_a,' = array A\n')
print(array_b,' = array B\n')
print(np.ravel(array_b),' = flattened array B\n')               # reduce from n-Dimensions to 1D array
print(array_a.reshape(2,4),' = new array A\n')                  # reshapes an array but does not change data
print(np.resize(array_a,(4,2)),' = new array A(numpy,resize)\n')# returns a new array with the size 4x2

# adding/removing elements 

array_1 = np.append(array_a,(3,8))                              # adding items(3,2) to array 1
print(array_a,' = array A\n')                                   # Display array A
print(array_1,' = array 1(append)\n')                           # Display array 1
array_1 = np.insert(array_1,7,100)                              # insert value 100 at position/element 7 in array_1
print(array_1,' = array 1(insert)\n')                           # Display array 1  
print(np.delete(array_1,[7]),' = array 1(delete)\n')            # deletes values of element 7 in array 1 


# In[16]:


# combining Arrays

array_a = np.array([[2,3],[4,5]])                    # create a 2x2 array A 
array_b = np.array([[9,8],[7,6]])                    # create a 2x2 array B

print(array_a,' = array A\n')                        # Display array A
print(array_b,' = array B\n')                        # Display array B

print(np.concatenate((array_a,array_b),axis=1),' = concatenated array\n') # concatenates array A and array B 
print(np.vstack((array_a,array_b)),' = Vstack(numpy)\n')                  # stacking arrays row wise - method 1
print(np.row_stack((array_a,array_b)),' = row stacking\n')                # stacking arrays row wise - method 2
print(np.r_[array_a,array_b],' = row stacking(r_)\n')                     # stacking arrays row wise - method 3


print(np.hstack((array_a,array_b)),' = Vstack(numpy)\n')                  # stacking arrays coloumn wise - method 1
print(np.column_stack((array_a,array_b)),' = coloumn stacking\n')         # stacking arrays coloumn wise - method 2
print(np.c_[array_a,array_b],' = coloumn stacking(c_)\n')                 # stacking arrays coloumn wise - method 3


# In[17]:


# splitting arrays

rand_array = np.random.randint(10,size=(3,6))                   # create a 3x6 array with random numbers
array_1 = np.copy(rand_array)                                             # copy rand_array to array_1

print(array_1,' = array 1\n')                                      # Display array 1
print(np.vsplit(array_1,1),' = vsplit array\n')                  # split the array vertically at the 1st index
print(np.hsplit(array_1,3),' = hsplit array\n')                  # split the array horizontlly at the 3rd index


# In[19]:


# saving and loading files

temp_array = np.random.randint(10,size=(6,6))                # create a random array and store it in temp
#array_a = np.copy(temp_array)                                # copy temp array to array A
np.savetxt('array_A.txt',array_a,delimiter=" ")              # save array A in a .txt file with delimiter as " "(space)
np.savetxt('array_A.csv',array_a,delimiter=',')              # save array A in a .csv file with delimiter as ',' (comma)
print(array_a,' = array A\n')                                # Display array A 

# loading files

array_b = np.loadtxt('array_A.txt')                          # load array_A.txt into array_b
array_c = np.genfromtxt('array_A.csv',delimiter=',')         # load values of array_A.txt into array_c 
print(array_b,' = array B(.txt file)\n')                     # Display array B (loaded from txt file)
print(array_c,' = array C(.csv file)\n')                     # Display array C (loaded from csv file)


# In[ ]:




