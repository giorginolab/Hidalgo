import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re

dir_name='/home/cristiano/polymer-dim-thesis/NONDEF/equilibrium/output/varia_N/dist/'
fs = [f for f in listdir(dir_name) 
    if isfile(join(dir_name, f))] 

for n in range(len(fs)):
    fs[n]=dir_name+fs[n]

#T=np.zeros(len(fs))
number1=np.zeros(len(fs))
number=np.zeros(len(fs))

'''
for i in range(len(fs)):
    result = re.search('dist/l(.*)_dist', fs[i]) 
    print(fs[i])
    #result= re.search('_(.*)_', result1.group(1))
    number[i]=result.group(1)
'''

for l in range(len(fs)):
    cmd= f'python test.py ' + fs[l] +' 1 3 10'
    #cmd= f'python test.py ' + fs[l] +' '+ str(int(number[l])) + ' 3 10'
    os.system(cmd)
    #print(number[l])
    
# in questo caso 1 seed per tutte le 9 temperature, poi rifaccio