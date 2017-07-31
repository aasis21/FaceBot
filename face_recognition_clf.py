"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""


from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib
import cv2



print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# geting database ready

data= np.load('data.npy')   

n_samples = data.shape[0]
X = data[1:, 0:data.shape[1]-1 ]  

n_features = data.shape[1] - 1

target = data[1:,data.shape[1]-1]
target_names = np.unique(target)
np.save('target',target_names)

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

X_train = X.astype(float)
for each in X_train:
    print(each.dtype)
y_train = target

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42)


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
joblib.dump(pca, "pca_clf.pkl", compress=3)
print("done in %0.3fs" % (time() - t0))



print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
x_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
clf = clf.fit(x_train_pca, y_train)
#clf = cv2.createFisherFaceRecognizer()
#clf.train(x_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
#print(clf.best_estimator_)

# Save the classifier
joblib.dump(clf, "recognition_clf.pkl", compress=3)




###############################################################################

# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
#print('score card:',clf.score(X_test_pca,y_test))
print("done in %0.3fs" % (time() - t0))






X_test = x_train_pca[21]
print("Predicting people's names on the test set")
y_pred = clf.predict(X_test)
print(y_pred)
print(target_names[np.int(y_pred[0])])
roi = X_train[21].reshape((150,150))
roi=np.array(roi, dtype=np.uint8)
plt.imshow(roi, cmap = 'gray', interpolation = 'bicubic')
plt.show()


X_test = x_train_pca[12]
print("Predicting people's names on the test set")
y_pred = clf.predict(X_test)
print(y_pred)
print(target_names)

roi = X_train[12].reshape((150,150))
roi=np.array(roi, dtype=np.uint8)
plt.imshow(roi, cmap = 'gray', interpolation = 'bicubic')
plt.show()


X_test = x_train_pca[51]
print("Predicting people's names on the test set")
y_pred = clf.predict(X_test)
print(y_pred)


roi = X_train[51].reshape((150,150))
roi=np.array(roi, dtype=np.uint8)
plt.imshow(roi, cmap = 'gray', interpolation = 'bicubic')
plt.show()
print(y_pred[0])







