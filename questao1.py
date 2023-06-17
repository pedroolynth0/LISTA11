import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


base = pd.read_csv('iris.csv', delimiter=',', encoding='cp1252')
Entrada = base.iloc[:, 0:4].values


scaler = MinMaxScaler()
Entrada = scaler.fit_transform(Entrada)


limit = int((Entrada.shape[0] // 2) ** 0.5)


silhouette_scores = []

for k in range(2, limit + 1):
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(Entrada)
    pred = model.predict(Entrada)
    score = silhouette_score(Entrada, pred)
    silhouette_scores.append(score)
    print('Silhouette Score k = {}: {:.3f}'.format(k, score))


wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=10)
    kmeans.fit(Entrada)
    wcss.append(kmeans.inertia_)




kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
num_clusters = kl.elbow


kmeans = KMeans(n_clusters=num_clusters, random_state=0)
saida_kmeans = kmeans.fit_predict(Entrada)


plt.scatter(Entrada[saida_kmeans == 0, 0], Entrada[saida_kmeans == 0, 1], s=100, c='purple', label='Iris-setosa')
plt.scatter(Entrada[saida_kmeans == 1, 0], Entrada[saida_kmeans == 1, 1], s=100, c='orange', label='Iris-versicolour')
plt.scatter(Entrada[saida_kmeans == 2, 0], Entrada[saida_kmeans == 2, 1], s=100, c='green', label='Iris-virginica')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.legend()
plt.title('Clustering Results (k = {})'.format(num_clusters))
plt.show()


plt.style.use("fivethirtyeight")
plt.plot(range(2, limit + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters Pedro O.C')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()
