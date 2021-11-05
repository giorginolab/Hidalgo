import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re

dir_name='/home/cristiano/polymer-dim-thesis/varia_T_def/dist/'
fs = [f for f in listdir(dir_name) 
    if isfile(join(dir_name, f))] 

for n in range(len(fs)):
    fs[n]=dir_name+fs[n]

T=np.zeros(len(fs))
number=np.zeros(len(fs))

for i in range(len(fs)):
    #result1 = re.search('num(.*)k', fs[i])
    #N[i]=result1.group(1)
    result2 = re.search('.0_(.*)_', fs[i]) #ricordo di dare temperature intere
    number[i]=result2.group(1)

for l in range(len(fs)):
    cmd= f'python test.py ' + fs[l] +' '+ str(int(number[l]))
    os.system(cmd)
    
# in questo caso 1 seed per tutte le 9 temperature, poi rifaccio