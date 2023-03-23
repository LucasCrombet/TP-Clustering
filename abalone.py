import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings(action="ignore")

column_names = ["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
abalone = pd.read_csv("abalone.data",names=column_names)
clean_up = {"Sex":{"I":0,"M":1,"F":2}}
abalone = abalone.replace(clean_up)
X = abalone

scale = StandardScaler()
X = scale.fit_transform(X)

n_clusters=30
cost=[]
for i in range(1,n_clusters):
    kmean= KMeans(i)
    kmean.fit(X)
    cost.append(kmean.inertia_)  
    
plt.plot(cost, 'bx-')
plt.show()

kmean= KMeans(5)
kmean.fit(X)
labels=kmean.labels_

clusters=pd.concat([abalone, pd.DataFrame({'cluster':labels})], axis=1)
clusters.head()

for c in clusters:
    grid= sns.FacetGrid(clusters, col='cluster')
    grid.map(sns.histplot, c)

dist = 1 - cosine_similarity(X)

pca = PCA(2)
pca.fit(dist)
X_PCA = pca.transform(dist)
X_PCA.shape

##############

x, y = X_PCA[:, 0], X_PCA[:, 1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow',
          4:"orange"}

names = {0: 'Principalement des enfants de petite taille et de poids très léger avec moins de 10 anneaux', 
         1: 'Principalemnt des males et femelles de grande taille et de poids moyen avec entre 8 et 20 anneaux', 
         2: 'Principalement des males et femelles de taille moyenne et de poids léger avec entre 8 et 20 anneaux', 
         3: '3:Principalement des males et femelles de très grande taille et de poids lourd avec entre 8 et 20 anneaux',
         4: "Principalement des enfants de taille moyenne et de poids léger avec entre 5 et 15 anneaux"}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Clustering avec k=5 en fonction des caractéristiques")
plt.show()

###########################"

kmean= KMeans(3)
kmean.fit(X)
labels=kmean.labels_

clusters=pd.concat([abalone, pd.DataFrame({'cluster':labels})], axis=1)
clusters.head()

for c in clusters:
    grid= sns.FacetGrid(clusters, col='cluster')
    grid.map(sns.histplot, c)

dist = 1 - cosine_similarity(X)

pca = PCA(2)
pca.fit(dist)
X_PCA = pca.transform(dist)
X_PCA.shape

##############

x, y = X_PCA[:, 0], X_PCA[:, 1]

colors = {0: 'red',
          1: 'blue',
          2: 'green'}

names = {0: 'Enfants de petite taille et de poids inférieur avec 10 anneaux ou moins', 
         1: 'Males/Femelles de taille moyenne et de poids moyen avec entre 8 et 20 anneaux', 
         2: 'Males/Femelles de grande taille et de poids supérieur avec entre 8 et 20 anneaux'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Clustering avec k=3 en fonction des caractéristiques")
plt.show()