# Importation des librairies
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd

# Chargement du dataset
column_names = ["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
abalone = pd.read_csv("abalone.data",names=column_names)

data = abalone[["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]]
y = abalone[["Rings"]]

# Elbow Method pour les K means# Importation de ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k est l'intervalle du nombre de clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(data)       
visualizer.show()        
# D'après la Elbow Method, le nombre k de de clusters optimal est 6
k = 6