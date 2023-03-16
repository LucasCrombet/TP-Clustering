# Importation des librairies
import numpy as np
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd

columns_names = ["sepal_length","sepal_width","petal_length","petal_width","class"]
iris_df = pd.read_csv("iris.data",names=columns_names)
