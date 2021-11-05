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
n_replica=int(sys.argv[2])
n_clusters=int(sys.argv[3])

#df=pd.read_csv(filename, sep=' ').head(6000).tail(1000)

# prove

#df = pd.DataFrame(1000000*np.random.rand(1000,10))
df1 = pd.DataFrame()
v=np.zeros(10000)
for a in range(100):
    for i in range(len(v)):
        v[i]=random.gauss(10.,4.)
    df1[str(a)]=v


#df1=(df.drop(['RG', 'n_contacts'], axis=1))

#df1= df.iloc[1: , :]

print(len(df1))
K_hid = n_clusters
#K_hid = 4
q_hid = 3 

#rho=3.*7.
#alpha=3.
#df_cont=np.pi/2-np.arctan((df-rho)/alpha)

X=np.asarray(df1)

############## run Hidalgo ##################################################################
n_iter=3000
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
#plt.scatter(df.n_contacts, model.V)
plt.show()


'''
result1 = re.search('dist/kbeta_(.*).0_', filename)
x=result1.group(1)
'''
'''
#with open('cart.csv', 'w', newline='') as output:
with open('ciao.csv', 'w', newline='') as output:
#with open('/home/cristiano/Hidalgo/python/out_hidalgo_equilibrium_beta/equi_beta_' + x + '_output_' + str(n_replica) + '.csv', 'w', newline='') as output:
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
