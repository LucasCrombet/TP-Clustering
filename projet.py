### EXERCICE 1 ###

#Q1

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#Q2
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], s=50)

#Q3


#Q4
plt.show()

#Q5
from sklearn.cluster import KMeans


#Q6
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X)

#Q7
y_kmeans = kmeans.predict(X)


#Q8
# La classe prédictée est la 3ème classe.


#Q9
#La classe prédictée est la 1ère classe.

#Q10

for elem in X:
    print(type(elem))


#Q11
print(y_kmeans)



### Exercice 2 ####

#Q1
import pandas as pd

#Q2
df = pd.read_csv("wine.csv")

#Q3
print(df.head(10))

#Q4
print(df[-10:-1])

#Q5
print(df.info())

#Q6
# Il n'y a pas de valeurs nulles

#Q7
df.isnull().sum()
# La méthode retourne None, donc il n'y a pas de valeurs nulles

#Q8
# Il y a des floats et des integers
print(df.columns)

#Q9
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Q10
donnees = df.to_numpy()

#Q11
pca = PCA(n_components=4)
donnees_APC = pca.fit_transform(donnees)

#Q12
from sklearn.cluster import KMeans

#Q13
kmeans = KMeans(n_clusters=3)

#Q14
classe = KMeans.fit_predict(X = donnees_APC,self=kmeans)

#Q15
labels_unique = np.unique(classe)


#Q16

# Ecrire une boucle sur les différentes valeurs de labels_unique, qui permettra d’afficher les nuages
# des points dans df correspondants à tous les labels



import matplotlib.pyplot as plt
# Obtenez toutes les valeurs uniques de labels dans df
labels_unique = df['labels'].unique()
# Boucle sur chaque valeur unique de labels
for label in labels_unique:
    # Sélectionnez les points de données correspondant à la valeur de label actuelle
    data = df[df['labels'] == label] 
    # Tracez le nuage de points pour les données actuelles
    plt.scatter(data['x'], data['y'], label=label)
# Ajoutez une légende pour les différents labels
plt.legend()
# Affichez le graphique final
plt.show()


#Q17
#plt.show()


### EXERCICE 3###

#Q1
import pandas as pd
import matplotlib.pyplot as plt

# Q2
df1 = pd.read_csv("World_hapiness_dataset_2019.csv")

# Q3
plt.scatter(df1["GDP per capita"],df1["Score"])
plt.show()

plt.scatter(df1["Healthy life expectancy"],df1["Freedom to make life choices"])
plt.show()

#Q4
import seaborn as sns

#Q5
sns.scatterplot(data = df1, x = "GDP per capita", y = "Score")

#Q6
sns.catplot(x="GDP per capita", y="Score", data=df1)