import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn 

n_punti=1000

length_phi = 10   #length of swiss roll in angular direction
length_Z = 13     #length of swiss roll in z direction
sigma = 0.01       #noise strength
m = n_punti         #number of samples

# create dataset
phi = length_phi*np.random.rand(m)
xi = np.random.rand(m)
Z = length_Z*np.random.rand(m)
X = 4./6*(phi + sigma*xi)*np.sin(phi)
Y = 4./6*(phi + sigma*xi)*np.cos(phi)

swiss_roll = np.array([X, Y, Z]).transpose()

# check that we have the right shape
print(swiss_roll.shape)
dfs=pd.DataFrame(swiss_roll, columns=list('xyz'))
dfs['Cluster']=(np.ones(n_punti)*3).astype(int)

#g = sns.PairGrid(dfs)
#g.map(sns.scatterplot)

#gau1 = pd.DataFrame(np.random.normal(0,1,size=(100, 5)), columns=list('ABCDE'))
gau2 = pd.DataFrame(np.random.normal(15,3,size=(n_punti, 3)), columns=list('xyz'))
gau1=pd.DataFrame()
gau1['Cluster']=(np.ones(n_punti)*2).astype(int)
gau=pd.concat([gau2, gau1.reindex(gau2.index)], axis=1)

x=np.add(np.linspace(-5,20,n_punti),np.random.normal(5,0.01,n_punti))
y=np.add(2*x-5,np.random.normal(5,0.01,n_punti))
z=np.random.normal(5,0.01,size=(n_punti, 1))
print(x.shape)
print(y.shape)
df=pd.DataFrame()
df['x']=x
df['y']=y
df['z']=z
df['Cluster']=np.ones(n_punti).astype(int)
ga=df.append(gau.append(dfs))
gaussmix=pd.DataFrame()
gaussmix= ga.reset_index(drop=True)
print(gaussmix)
#g=sns.PairGrid(gaussmix, hue='species',  hue_kws=dict(marker="s", linewidth=1))
#g.map(sns.scatterplot)

gaussmix.to_csv('MIX.csv')

sns.pairplot(
    gaussmix, hue='Cluster',
    plot_kws=dict(marker="+",s=1., linewidth=1),
    diag_kws=dict(fill=False),
)

plt.show()
