
# coding: utf-8

# In[1]:


import pandas as pd               # for data manipuliation         
import IPython                    # for displaying the data structures
import numpy as np                # for array manipulation
import matplotlib.pyplot as plt   # for data visulization


# In[2]:


# Series

s = pd.Series([5,7,8,2,3],index=['a','b','c','d','e'])         # 1D labelled array capable of holding any data types
display(s)


# In[3]:


# Dataframe
data = {'Country' : ['india','france','england'],
       'Capital' : ['New dehli','Paris','landon'],
       'Population' : [134212015,111908046,204565789]}
df = pd.DataFrame(data,columns=['Country','Capital','Population'])
display(df)


# In[4]:


# input and ouput

# read and write csv
df.to_csv('country.csv',index=None)             # Creates a csv file with values of df and no index
df1 = pd.read_csv('country.csv')                # reads values from country.csv and saves it to df1
display(df1)                                    # display df1 

# read and write from excel
df.to_excel('country.xlsx',sheet_name='country')            # writes values of df to counrty.xlsx
df2 = pd.read_excel('country.xlsx',sheetname='country')     # reads values from country.xlsx and saves it to df2
display(df2)                                                # display df2 


# In[5]:


# selection

# getting elements
display(s)                                        # display array 
print(s['b'],' = value of b\n')                   # get one element

display(df)                                       # display dataframe
print(df[1:],' = subset of df\n')                 # get subset of dataframe

# by position
print(df.iloc[0,0],' = element at 0th row and 0th coloumn\n')                          # select single value by row and coloumn
print(df.iat[2,0],' = element at 2nd row and 0th coloumn\n')                           # select element at sepcifiedlocation

# by label
print(df.loc[0,'Country'],' = element at 0th row and under the coloumn of country\n')  # select single value by row and coloumn labels
print(df.at[1,'Country'],' = element at 1st row and under the coloumn of country\n')   # select single value by row and coloumn labels

# by label 
print(df.ix[2],' = select row 2\n')                                                    # select single row of subset of rows
print(df.ix[:,'Capital'],' = select capital coloumn\n')                                # select single coloumn of subset of coloumn 
print(df.ix[1,'Capital'],' = select 1 st row element in capital\n')                    # select rpw and coloumn

# boolean indexing 
print(df[df['Population']>12000000],' = population > 120000000\n')                     # filter to adjust dataframe
print(s,'\n')                                                                          # display s
print(s[~(s>3)],' = elements graeter than 3\n')                                        # series where value is not > 3    
print(s[(s < 1) | (s > 4)],'= <1 or >4\n')                                             # s value where s is < 1 or > 4 

# setting
s['a'] = 10                                                                            # set index value a to 10 
display(s)


# In[6]:


# Droping
display(s)                                       # display s
s1 = s.drop(['a','c'])                           # drop values from rows (axis = 0) default
print(s1,' = s1(dropped values)')                # display dropped values

display(df)                                      # display df
df.drop('Country',axis=1)                        # drop values from coloumns (axis = 1) 


# In[7]:


# sorting and ranking
display(df)                                                            # display df
print(df.sort_index(),' = sorted by labels\n')                         # sort by labels along an axis
print(df.sort_values(by='Country'),' = sorted by counrty\n')           # sort by values along an axis
print(df.rank(),' = ranks\n')                                          # assign ranks to entries


# In[8]:


# Data frame and Series information

# basic information
display(df)                                               # display df 
print(df.shape,' = shape of df\n')                        # display shape of df (rows,coloumns) 
print(df.index,' = index\n')                              # describe index
print(df.columns,' = coloumns\n')                         # describe dataframe coloumns
print(df.info(),' = info\n')                              # returns info on df
print(df.count(),' = count\n')                            # returns number of non-NA values

# Summary
print(df.sum(),' = sum\n')                                # returns sum of values
print(df.cumsum(),' = cummulative sum\n')                 # returns cummulative sum of df
print(df.min(),' = min value\n')                          # returns the minimum value of df
print(df.max(),' = max value\n')                          # returns max value
print(df.describe(),' = description of df\n')             # returns summary statistics
print(df.mean(),' = mean\n')                              # returns the mean of df
print(df.median(),' = median\n')                          # returns median of values in df


# In[9]:


# applying functions

display(df)                                       # display dataframe df
f = lambda x: x*2                                 # define function
print(df.apply(f),' = applied function f\n')      # apply the function f
print(df.applymap(f),' = apply map\n')            # apply function f element wise


# In[10]:


# Data allignment

# internal data alignment
s2 = pd.Series([3,9,5],index=['a','c','d'])                        # create a new series
print(s2,' = s2\n')                                                # display s2
print(s,' = s\n')                                                  # display s  
s3 = s + s2                                                        # NA values are introduced in the indices that dont overlap 
print(s3,' = s3(s + s2)\n')                                        # display s3


# arithemetic operations with fill method
print(s.add(s3,fill_value=0),' = add\n')                                        # display s3 with values after adding with s, any NaN will be considered as 0
print(s.sub(s3,fill_value=2),' = sub\n')                                        # display s3 with values after substracting with s, any NaN will be considered as 0
print(s.mul(s3,fill_value=4),' = mul\n')                                        # display s3 with values after multiplying with s, any NaN will be considered as 0
print(s.div(s3,fill_value=3),' = div\n')                                        # display s3 with values after dividing with s, any NaN will be considered as 0


# In[11]:


# reshaping data

data = {'date' : ['2017-03-01','2017-02-01','2017-04-01','2017-01-01','2017-06-01','2017-05-01'],
       'type' : ['a','b','a','c','d','b'],
       'value' : [11.3,12.6,96.2,14.3,12.5,45.3]}                                   # set new values for dataframe
df3 = pd.DataFrame(data,columns=['date','type','value'])                            # create new dataframe
print(df3,' = df3\n')                                                               # display dataframe

df4 = df3.pivot(index='date',columns='type',values='value')                         # spread rows into coloumns
print(df4,' = df4\n')                                                               # display the pivoted df  
 
df5 = pd.pivot_table(df3,values='value',index='date',columns='type')                # spread rows into coloumns
print(df5,' =df5\n')                                                                # display df5


# In[12]:


# Stack & unstack
print(df4,' = df4\n')                                               # display df4
stacked = df4.stack()                                               # pivot a level of coloumn labels
print(stacked,' = stacked\n')                                       # display stacked
print(stacked.unstack(),' = unstacked')                             # pivote a level of index labels


# In[13]:


# melt

display('df3',df3)                                                                             # display df3
df6 = pd.melt(df3,id_vars=['date'],value_vars=['type','value'],var_name='observations')        # gather coloumns into rows 
display('df6',df6)                                                                             # display df6  


# In[14]:


# advanced indexing

display('df3',df3)                                                # display df3
#selecting
print(df3.loc[:,(df3>1).any()],' = values > 1(any)\n')            # select coloumns with any values > 1
print(df3.loc[:,(df3>1).all()],' = values > 1(all)\n')            # select coloumns with values > 1
print(df3.loc[:,df3.isnull().any()],' = NaN values\n')            # select coloumns with NaN
print(df3.loc[:,df3.notnull().all()],' = non-NaN values\n')       # select coloumns without NaN values

# indexing with isin
print(df3.filter(items=['a','b']),' = filtered values\n')         # filter on values
display('df',df)                                                  # display df
print(df[df.Country.isin(df2.Country)],' = same elements(isin)\n')# find same elements
print(df.select(lambda x: not x%5),' = select\n')                 # find specific elements 

# where
display('s',s)                                                    # display S
print(s.where(s > 4),' = s > 4(where)\n')                         # subset the data


# In[15]:


# setting and resetting index
display('df',df)                                              # dispaly df    
df7 = df.set_index('Capital')                                 # set the index (here set it to Capital)
display('df7',df7)                                            # display df7

df8 = df.reset_index()                                        # reset the index
display('df8',df8)                                            # display df8
df7 = df7.rename(index=str,columns={'Country':'cntry',
                                   'Capital':'cptl',
                                   'Population':'ppltn'})     # rename dataframe 
display('df7',df7)                                            # display ddf7 


# In[16]:


# reindexing
display('s',s)                                  # display s
s4 = s.reindex(['c','e','d','a','b'])           # reindexing s with new values
display('s4',s4)


# In[17]:


# multi indexing

arrays = [np.array([1,2,3]),
         np.array([7,6,5])]                                   # create 3x2 array
df9 = pd.DataFrame(np.random.rand(3,2),index=arrays)          # create dataframe df9 
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples,names=['first','second'])    # create an index 
df10 = pd.DataFrame(np.random.rand(3,2),index=index)                  # create dataframe with created index
display('df10',df10)                                                  # display df10 
display('df3',df3)                                                    # display index df3
display('df3(index)',df3.set_index(['date','type']))                  # change and display index


# In[18]:


# dupicate data
display('s',s)                                          # display S
print(s.unique(),' = unique values(s)\n')               # return unique values
display('df3',df3)                                      # display df3
print(df3.duplicated('type'),' = dupicated(df3)\n')     # check dupicates in coloumn type in df3
df11 = df3.drop_duplicates('type',keep='last')          # drop dupicates
display('df11',df11)                                    # display df11
display('df',df)                                        # display df
print(df.index.duplicated(),' = dupicated(df)\n')       # check index duplicates


# In[19]:


# missing data

display('s3',s3)                                   # display s3
s5 = s3.dropna()                                   # drop NaN values
print(s5,' = dropped values(s5)\n')                # display s5
s6 = s3.fillna(s3.mean())                          # fill NaN(Not a Number) with a predetermined value
print(s6,' = filled values(s6)\n')                 # display s6
s7 = s3.replace('d','a')                           # replace values with others
print(s7,' = replaced values(s7)\n')               # display s7


# In[20]:


# combining data 
X = {'X1' : ['a','b','c'],
    'X2' : [12,55,34]}                                        # create a dataset
data1 = pd.DataFrame(X,columns=['X1','X2'])                   # create dataframe data1
display('data1',data1)                                        # display dataframe data1

X2 = {'X1' : ['a','b','d'],
     'X3' : [56,12,75]}                                       # create a dataset
data2 = pd.DataFrame(X2,columns=['X1','X3'])                  # create a dataframe data2
display('\ndata2',data2)                                      # display dataframe data2

display('\nRight',pd.merge(data1,data2,how='right',on='X1'))  # merge data1 with data2 along X1 on the right
display('\nleft',pd.merge(data1,data2,how='left',on='X1'))    # merge data1 with data2 along X1 on the left
display('\ninner',pd.merge(data1,data2,how='inner',on='X1'))  # merge data1 with data2 along X1 on the inner
display('\nouter',pd.merge(data1,data2,how='outer',on='X1'))  # merge data1 with data2 along X1 on the outer


# In[21]:


# joining 

display('data1',data1)                               # display data1
display('data2',data2)                               # display data2
data3 = data1.join(data2,lsuffix='X1')               # with X1 as a base join data1 and data2
display('data3(join)',data3)                         # display data3

pd.concat([data1,data2],axis=1,join='inner')         # join data1 and data2 


# In[22]:


# concatenate
display('s',s)                         # display s
display('\ns2',s2)                     # display s2
s8 = s.append(s2)                      # append s2 on s vertically
display('\ns8(appended)',s8)           # display s8 appended


# In[23]:


# dates
display('df3',df3)                                        # display df3
df3['date'] = pd.to_datetime(df3['date'])
df3['date'] = pd.date_range('2017-01-01',periods=6,freq='M')
index = pd.DatetimeIndex((2012,1,2))
display(df3)


# In[24]:


# visualization

display('s',s)                          # display s
s.plot()                                # plot s
plt.show()                              # display graph

display('\ndf3',df3)                    # display df3
df3.plot()                              # plot df3
plt.show()                              # display graph


# In[ ]:




