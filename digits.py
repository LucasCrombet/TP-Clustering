# Importation des librairies
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Chargement du dataset
digits = load_digits()
data = scale(digits.data)
y = digits.target



# Elbow Method pour les K means# Importation de ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k est l'intervalle du nombre de clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(digits.data)       
visualizer.show()        

# D'après la Elbow Method, le nombre k de de clusters optimal est 9
k = 19

# Score
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
    

# Entrainement du modèle
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

# Visualisation avec matplotlib

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))

sample_size = 300

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Taille du saut du mesh. Diminuer pour augmenter la qualité.
h = .02     # point dans le mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtient des labels pour chaque point
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Transforme le resultat en un plot en couleur
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot les centres
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-Means Clustering du dataset digits')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()