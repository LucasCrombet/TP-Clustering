from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits

digits = load_digits()
x = digits.data

# Elbow Method for K means# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(x)        # Fit data to visualizer
visualizer.show()        # Finalize and render figure

kmeans = KMeans(n_clusters=9)
classe = kmeans.fit_predict(X = x)

plt.scatter(
    x[classe == 0, 0], x[classe == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='cluster 1')


plt.legend(scatterpoints=1)
plt.grid()
plt.show()
