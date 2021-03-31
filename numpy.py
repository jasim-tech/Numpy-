#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
arr=np.array([1,2,3,4,5])
print(arr)


# In[7]:


import numpy as np
arr=np.array([1,2,3,4,5])
print(type(arr))


# In[9]:


arr=np.array([[1,2,3] ,[4,5,6]])
print(arr)


# In[11]:


arr=np.array([[[1,2,3] ,[4,5,6]],[[23,5,7],[4,6,8]]])
print(arr)
print(arr.ndim)


# In[12]:


import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)


# In[16]:


#Create an array with 5 dimensions and verify that it has 5 dimensions:
arr=np.array([1,2,3],ndmin=5)
print(arr)
print("dimension is:",arr.ndim)


# In[13]:


import numpy as np
arr=np.array([[1,43,5,8,5,98,87],[21,2,34,45,56,67,98]],dtype='int16')
print(arr)


# In[3]:


#get dimension
arr.ndim


# In[4]:


#get shape of matrix
arr.shape


# In[14]:


#Get type
print(arr.dtype)


# In[8]:


#Get size
print(arr.itemsize)


# In[15]:


#Get total size
arr.nbytes


# In[16]:


#get specific item[r,c]
print(arr[1,2])


# In[21]:


#get specific row
print(arr[1,:])
#get specific column
print(arr[:,3])


# In[22]:


#change an specific item
arr=np.array([[1,43,5,8,5,98,87],[21,2,34,45,56,67,98]],dtype='int16')
print(arr)
arr[0,1]=2
print(arr)


# In[32]:


#Get item with step index
arr[0,0:7:2]


# In[35]:


arr[0,-7:-1:2]


# In[37]:


#change the column 
arr[:,1]=[2,22]
print(arr)


# In[38]:


#change the row
arr[0,:]=[98,87,76,54,43,32,21]
print(arr)


# In[39]:


#3-D array
import numpy as np
arr=np.array([[[1,2],[2,3]],[[3,4],[4,5]]])
print(arr)


# In[41]:


#Get specific element in 3-D array
print(arr[1,1,1])
print(arr[0,1,0])


# In[43]:


#Get specific row
print(arr[0,0,:])


# In[45]:


#Get specific column
print(arr[1,:,1])


# In[46]:


#change the specific item and specific row or column
arr[0,1,1]=33
print(arr)


# In[47]:


#chnage row
arr[1,1,:]=[33,44]
print(arr)


# In[48]:


#change column
arr[0,:,0]=[11,22]


# In[8]:


import numpy as np
np.zeros((3,4,4,2))


# In[6]:


import numpy as np
np.full((3,3,2),99)
a=np.array([[[1,2],[2,3]],[[1,2],[2,3]]])
print(a)


# In[11]:


np.full_like(a,4)


# In[8]:


#random
np.random.rand(4,2)


# In[10]:


#random array as like a 
np.random.random_sample(a.shape)


# In[16]:


#random array with integer
np.random.randint(-2,3,size=(3,3))


# In[17]:


#identity matrix
np.identity(3)


# In[27]:


#repeat an array 
arr=np.array([[[[1,2,3,4]]]])
r1=np.repeat(arr,4,axis=2)
print(r1)


# In[41]:


x=np.ones((6,6))
print(x)
y=np.zeros((4,4))
print(y)


y[(1,1)]=9
y[2,2]=8
print(y)
x[1:5,1:5]=y
print(x)


# In[5]:


import numpy as np
a=np.array([1,2,3,4])
print(a)
print(a+2)
print(a-2)
print(a*2)
print(a**2)
print(a/2)
b=np.array([0,1,0,1])
print(a+b)
print(np.sin(a))
print(np.cos(a))


# Linear algebra

# In[7]:


import numpy as np
a=np.ones((3,4))
print(a)
b=np.full((4,3),3)
print(b)
output=np.matmul(a,b)
print(output) 


# Statistics

# In[16]:


a=np.array([[1,2,3],[4,5,6]])
print(a)
print(np.max(a))
print(np.min(a)) 
print(np.max(a,axis=0))
print(np.max(a,axis=1))
print(np.min(a,axis=0))
print(np.min(a,axis=1))
print(np.sum(a))
print(np.sum(a,axis=0))
print(np.sum(a,axis=1))


# Recongnizing array

# In[21]:


before=np.array([[1,3,5,7],[2,4,6,8]])
print(before)
after=before.reshape(4,2)
print(after)


# In[24]:


after=before.reshape(2,2,2)
print(after)


# In[25]:


print(before)


# In[32]:


#Vertical stacking vector
import numpy as np
v1=np.full((2,2),2)
v2=np.full((2,2),3)
ver=np.vstack([v1,v2,v1])  
print(ver)


#harizontal stacking vctor
har=np.hstack([v1,v2,v1])
print(har)


# Miscellenous

# In[47]:


import numpy as np
filedata=np.genfromtxt('DATA.txt',delimiter=',')
filedata=filedata.astype('int32')
print(filedata)


# In[42]:


filedata>3


# In[43]:


filedata[filedata>9]


# In[46]:


a=np.array([1,2,3,4,5,6,7,8])
a[[1,3,5]]


# In[53]:


((filedata<20) & (filedata>100))


# In[ ]:




