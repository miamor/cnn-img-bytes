import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler



features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
)

print('features[:5]', features[:5])
print('true_labels[:5]', true_labels[:5])



scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print('scaled_features[:5]', scaled_features[:5])


kmeans = KMeans(
    init="random",
    n_clusters=2,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(scaled_features)


print('kmeans.cluster_centers_', kmeans.cluster_centers_)
print('kmeans.labels_[:5]', kmeans.labels_[:5])




