
# coding: utf-8

# In[85]:


import pandas as pd
import matplotlib. pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


# In[86]:


M= 20
N = 20
O = 20


# In[151]:


rhs_3d = "rhs_3d.txt"
rhs = "rhs.txt"
RHS = [rhs_3d, rhs]
tmp_3d = "tmp.txt"
tmp = "_tmp.txt"
TMP = [tmp_3d, tmp]
d_r_3d = "d_r_3d.txt"
d_r = "d_r.txt"
DR = [d_r_3d, d_r]
d_result_3d = "d_result_3d.txt"
d_result = "d_result.txt"
DRES = [d_result_3d, d_result]

TYPE = [RHS, TMP, DR, DRES]


# In[152]:


for T in TYPE:
    result_3d = []
    file_in = open(T[0], 'r')
    for y in file_in.read().split('\n'):
        if y.rstrip("\n"): 
            #print(y)
            result_3d.append(float(y))


    # In[154]:


    result = []
    file_in = open(T[1], 'r')
    for y in file_in.read().split('\n'):
        if y.rstrip("\n"): 
            #print(y)
            result.append(float(y))




    v = np.asarray(result)
    v3d = np.asarray(result_3d)
    #v3d2 = np.asarray(result_3d_2)


    # In[149]:
    
    print(T[1], T[0])
    print("%6.3f, %6.3f, %6.3f" % (v.mean(), v.max(), v.min()))
    print("%6.3f, %6.3f, %6.3f"  % (v3d.mean(), v3d.max(), v3d.min()))

