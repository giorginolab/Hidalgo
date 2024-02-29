import numpy as np
from dimension import TwoNN
from dimension import hidalgo
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
import random

filename = str(sys.argv[1])
onetwothree=int(sys.argv[2])
n_clusters=int(sys.argv[3])
n_replicas=int(sys.argv[4])


df1=pd.read_csv(filename, sep=',', header=0)#.tail(1000) #occhio qua...
#df1=(df.drop(['RG', 'n_contacts'], axis=1))

print(df1.columns)

df1=(df1.drop(['Unnamed: 0', 'Cluster'], axis=1))
#plt.plot(df1.iloc[:,0])
#plt.show()

print('n_punti: ', len(df1))
K_hid = n_clusters
q_hid = 3 

################
print(len(df1))
print('quanti nan: ',df1.isnull().sum().sum())

#rows_with_nan = []
#for index, row in df.iterrows():
#    is_nan_series = row.isnull()
#    if is_nan_series.any():
#        rows_with_nan.append(index)

#print(rows_with_nan)
###############

X=np.asarray(df1)

############## run Hidalgo ##################################################################
n_iter=5000
burnin=0.10

#model = hidalgo(K=K_hid, Niter=n_iter, q=q_hid, burn_in=burnin, zeta=0.75) 
model=hidalgo(K=K_hid,Niter=n_iter,zeta=0.75,q=q_hid,Nreplicas=n_replicas,burn_in=burnin)

model.fit(X)
print('d: ',model.d_,', derr: ', model.derr_) #ID of each of the clusters

# plotting likelihood (k)
'''
with open('likelihood.csv', 'a', newline='') as output:
#with open('/home/cristiano/Hidalgo/python/out_hidalgo_equilibrium_beta/equi_beta_' + x + '_output_' + str(onetwothree) + '.csv', 'w', newline='') as output:
    output.write(str(K_hid) + ' ' + str(model.lik_)) #ID of each of the clusters
    output.write('\n')
'''

plt.plot(model.V, marker='o', markersize=0, markeredgecolor='black', markeredgewidth=0.1, linewidth=1)
#plt.scatter(df.n_contacts, model.V)
plt.show()
print('media: ',np.mean(model.V))
print('mediana: ',np.median(model.V))

'''
result= re.search('dist/l_(.*)_rep', filename)
#xa=result.group(1)
#result= re.search('(.*).0', xa)
x=result.group(1)
result1 = re.search('rep_(.*)_dist', filename)
x1=result1.group(1)
print(x)
print(x1)

with open('/home/cristiano/Hidalgo/python/output/l/newone_l_' + x + '_'+ x1+ '.csv', 'w', newline='') as output:
    output.write('Mean Median StdDev \n')
    for i in range(len(model.V)):
        output.write(str(model.V[i])+ ' ')
        output.write(str(model.Vmedian[i])+ ' ')
        output.write(str(model.Vstd[i]))
        output.write('\n')
'''

'''
D = euclidean_distances(X)

model = hidalgo(metric="predefined", K=K)

model.fit(D)

print(model.d_, model.derr_)
print(model.p_, model.perr_)
print(model.lik_, model.likerr_)
print(model.Pi)
print(model.Z)
'''
