import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def davies_bouldin_index(X, labels, centroids):
    num_clusters = len(np.unique(labels))
    distances = pairwise_distances(X, centroids)
    cluster_distances = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_distances[i] = np.mean(distances[labels == i, i])
    
    db = 0.0
    for i in range(num_clusters):
        max_distance = 0.0
        for j in range(num_clusters):
            if i != j:
                distance = (cluster_distances[i] + cluster_distances[j]) / distances[i, j]
                if distance > max_distance:
                    max_distance = distance
        db += max_distance
    
    db /= num_clusters
    
    return db


base = pd.read_csv('iris.csv', delimiter=',', encoding='cp1252')
Entrada = base.iloc[:, 0:4].values
Classes = base.iloc[:, 4].values
gt = base['class'].values


scaler = MinMaxScaler()
Entrada = scaler.fit_transform(Entrada)


limit = int((Entrada.shape[0] // 2) ** 0.5)


silhouette_scores = []
davies_bouldin_scores = []

for k in range(2, limit + 1):
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(Entrada)
    pred = model.predict(Entrada)
    silhouette = silhouette_score(Entrada, pred)
    db = davies_bouldin_index(Entrada, pred, model.cluster_centers_)
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(db)
    print('k = {}, Silhouette Score: {:.3f}, Davies-Bouldin Index: {:.3f}'.format(k, silhouette, db))


wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=10)
    kmeans.fit(Entrada)
    wcss.append(kmeans.inertia_)


plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), wcss)
plt.xticks(range(2, 11))
plt.title('The elbow method - Pedro O.C.')
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
num_clusters = kl.elbow


kmeans = KMeans(n_clusters=num_clusters, random_state=0)
saida_kmeans = kmeans.fit_predict(Entrada)


plt.scatter(Entrada[saida_kmeans == 0, 0], Entrada[saida_kmeans == 0, 1], s=100, c='purple', label='Iris-setosa')
plt.scatter(Entrada[saida_kmeans == 1, 0], Entrada[saida_kmeans == 1, 1], s=100, c='orange', label='Iris-versicolor')
plt.scatter(Entrada[saida_kmeans == 2, 0], Entrada[saida_kmeans == 2, 1], s=100, c='green', label='Iris-virginica')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')

plt.legend()
plt.title('Clustering Results (k = {})'.format(num_clusters))
plt.show()


plt.plot(range(2, limit + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()


plt.plot(range(2, limit + 1), davies_bouldin_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index vs Number of Clusters')
plt.show()

plt.scatter(Entrada[gt == 'Iris-setosa', 0], Entrada[gt == 'Iris-setosa', 1], s=100, c='purple', label='Iris-setosa')
plt.scatter(Entrada[gt == 'Iris-versicolor', 0], Entrada[gt == 'Iris-versicolor', 1], s=100, c='orange', label='Iris-versicolor')
plt.scatter(Entrada[gt == 'Iris-virginica', 0], Entrada[gt == 'Iris-virginica', 1], s=100, c='green', label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.legend()
plt.title('True Clusters')
plt.show()