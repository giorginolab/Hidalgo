import numpy as np
from dimension import TwoNN
from dimension import hidalgo
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
import random

#filename = str(sys.argv[1])
#n_replica=int(sys.argv[2])
n_clusters=int(sys.argv[1])
punti=int(sys.argv[2])
filled_cols=int(sys.argv[3])
#df=pd.read_csv(filename, sep=' ').head(6000).tail(1000)

# prove

#df = pd.DataFrame(1000000*np.random.rand(1000,10))
#gau = pd.DataFrame()
#uni = pd.DataFrame()
#v=np.zeros(10000)

#for a in range(150):
#    for i in range(len(v)):
#        v[i]=random.gauss(10.,4.)
    #uni[str(a)]=np.random.uniform(len(v))
    #gau[str(a)]=v



gau1 = pd.DataFrame(np.random.normal(0,1,size=(punti, filled_cols)))
gau2 = pd.DataFrame(np.zeros(shape=(punti, 150-filled_cols)))
gau=pd.concat([gau1, gau2.reindex(gau1.index)], axis=1)
uni1 = pd.DataFrame(np.random.uniform(0,10,size=(punti, filled_cols)))
uni2 = pd.DataFrame(np.zeros(shape=(punti, 150-filled_cols)))
uni=pd.concat([uni1, uni2.reindex(uni1.index)], axis=1)
#uni=pd.concat(uni1,uni2)
#df1=(df.drop(['RG', 'n_contacts'], axis=1))
#print(uni)
#df1= df.iloc[1: , :]

#print(len(gau), len(uni))

K_hid = n_clusters
#K_hid = 4
q_hid = 3 

X=np.asarray(gau)
#X=np.asarray(uni)

############## run Hidalgo ##################################################################
n_iter=100000
#n_iter=50000 #right
burnin=0.10
model = hidalgo(K=K_hid, Niter=n_iter, q=q_hid, burn_in=burnin, zeta=0.75) 

# model=hidalgo(K=2,Niter=2000,zeta=0.65,q=5,Nreplicas=10,burn_in=0.8)


model.fit(X)
print('d: ',model.d_,', derr: ', model.derr_) #ID of each of the clusters


'''
with open('likelihood.csv', 'a', newline='') as output:
#with open('/home/cristiano/Hidalgo/python/out_hidalgo_equilibrium_beta/equi_beta_' + x + '_output_' + str(n_replica) + '.csv', 'w', newline='') as output:
    output.write(str(K_hid) + ' ' + str(model.lik_)) #ID of each of the clusters
    output.write('\n')
'''

plt.plot(model.V, marker='o', markeredgecolor='black', markeredgewidth=0.1, linewidth=0)
plt.show()

'''
with open('prove_gau/2000/gau_colonne_' + str(filled_cols) + '.csv', 'w', newline='') as output:
    output.write('Mean Median StdDev \n')
    for i in range(len(model.V)):
        output.write(str(model.V[i])+ ' ')
        output.write(str(model.Vmedian[i])+ ' ')
        output.write(str(model.Vstd[i]))
        output.write('\n')
'''