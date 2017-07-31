from sklearn.cluster import AffinityPropagation, KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn.decomposition import PCA
from  time import time
import matplotlib.pyplot as plt
import cv2


data= np.load('data.npy')

n_samples = data.shape[0]
X = data[:, 0:data.shape[1]-1 ]

n_features = data.shape[1] - 1

target = data[:,data.shape[1]-1]
target_names = np.unique(target)
print(target_names)
print(target_names.shape)
print(target_names[2])
for i,person in enumerate(target):
    idx = np.where(target_names==person)
    print(idx[0][0])
    target[i]=idx[0][0]


print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
#X_train, X_test, y_train, y_test = train_test_split(
   # X, y, test_size=0.25, random_state=42)
data= np.empty(150*150)

for each in X:
    roi = each.reshape((150, 150, 3))
    roi = np.array(roi, dtype=np.uint8)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    roi_data = gray.ravel()
    data = np.vstack((data, roi_data))

y_train = target
X_train = X.astype(float)


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(data)
print("done in %0.3fs" % (time() - t0))



print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
x_train_pca = pca.transform(data)

print("done in %0.3fs" % (time() - t0))




print('finding no of cluter')
af = AffinityPropagation(preference=-50).fit(x_train_pca)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)

print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels, metric='sqeuclidean'))
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))


clusterer = KMeans(n_clusters= 50, random_state=10)

cluster_labels = clusterer.fit_predict(X_train)
print(cluster_labels)


print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X,cluster_labels, metric='sqeuclidean'))
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X,cluster_labels))
cluster_labels = clusterer.fit_predict(x_train_pca)
for i, label in enumerate(cluster_labels):
    if label == 24:
        X_test = X_train[i]
        print("Predicting people's names on the test set")

        roi = X_test.reshape((150, 150, 3))
        roi = np.array(roi, dtype=np.uint8)
        plt.imshow(roi, cmap='gray', interpolation='bicubic')
        plt.show()
