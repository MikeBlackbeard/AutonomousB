#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import os


# In[3]:


os.getcwd()


# In[5]:


file = open ("C:\\Users\\ACER\\Desktop\\HSHL Online\\Semester 7\\Autonomous lab\\Task2.txt", "r")


# In[6]:


print("The output of this file is")
print(file.read())


# In[7]:


file.close() #change from reading to writing


# In[8]:


os.getcwd()


# In[9]:


file1 = open ("C:\\Users\\ACER\\Desktop\\HSHL Online\\Semester 7\\Autonomous lab\\Task2.txt", "w")


# In[10]:


Line = ["\n This is the second task of autonomous lab \n\n\n The task of reading and writing is done\n"]


# In[12]:


file1.writelines(Line)
file1.write("Bye!!!!\n")


# In[13]:


file1.close()


# In[15]:


file1 = open ("C:\\Users\\ACER\\Desktop\\HSHL Online\\Semester 7\\Autonomous lab\\Task2.txt", "r")


# In[16]:


print("The added part is")
print(file1.read())


# In[ ]:




