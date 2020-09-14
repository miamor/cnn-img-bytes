import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

K = 2

# %% Initialisation
# features, true_labels = make_blobs(
#     n_samples=200,
#     centers=K,
#     cluster_std=2.75,
#     random_state=42
# )
raw_features = np.load('data_prepared/train__X_rgb.npy')
true_labels = np.load('data_prepared/train__Y.npy')

# pass through model to extract features
model_name = 'rgb'
model = load_model('models/'+model_name+'.h5')
model.summary()
intermediate_layer_model = Model(inputs=model.input, 
                                 outputs=model.get_layer("leaky_re_lu_3").output)
features = intermediate_layer_model.predict(raw_features)

print('raw_features.shape', raw_features.shape)
print('features.shape', features.shape)


features = features.reshape((features.shape[0], -1)) / 255.0
true_labels = true_labels.ravel()
print('features.shape', features.shape)
print('true_labels.shape', true_labels.shape)
# print('features[:5]', features[:5])
# print('true_labels[:5]', true_labels[:5])


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=K)
kmeans.fit(features)

# Then we learn the labels
labels = kmeans.predict(features)
centroids = kmeans.cluster_centers_
# print('labels', labels)
# print('true_labels', true_labels)
print('labels', labels.shape)


# Plot
colmap = {0: 'g', 1: 'r'}
shapemap = {0: 'o', 1: 'x'}

fig = plt.figure(figsize=(8, 8))

ft0 = np.mean(features, axis=1)
# print('ft0', ft0)
# idx_bgn = true_labels == 0
# idx_mal = true_labels == 1
# colors_bgn = list(map(lambda x: colmap[x], labels[idx_bgn]))
# colors_mal = list(map(lambda x: colmap[x], labels[idx_mal]))
idx_bgn = labels == 0
idx_mal = labels == 1
colors_bgn = list(map(lambda x: colmap[x], true_labels[idx_bgn]))
colors_mal = list(map(lambda x: colmap[x], true_labels[idx_mal]))
colors = list(map(lambda x: colmap[x], true_labels))

# plt.scatter(ft0[idx_bgn], labels[idx_bgn], color=colors_bgn, alpha=0.5, marker='o')
# plt.scatter(ft0[idx_mal], labels[idx_mal], color=colors_mal, alpha=0.5, marker='x')
plt.scatter(ft0, labels, color=colors, alpha=0.5, marker='o')

# for idx, centroid in enumerate(centroids):
#     plt.scatter(*centroid, color=colmap[idx], marker='o')

plt.xlim(0, 1)
plt.ylim(-0.2, 1.2)
plt.show()