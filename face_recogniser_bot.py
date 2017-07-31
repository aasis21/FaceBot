# Import the modules
import cv2
from sklearn.externals import joblib
import numpy as np
from matplotlib import pyplot as plt
import glob

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
import os

    
print(__doc__)

# classifier to identify face in image using opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


data= np.empty(150*150)

# convering png image in to jpg format
pngs = glob.glob('test_images/*.png')
for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img) 
    os.remove(j)


test_images = glob.glob("test_images/*.jpg")
for image in test_images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detecting all faces in image..
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        roi_color = img[y:y+h, x:x+w]

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (150, 150), interpolation=cv2.INTER_AREA)
       
        roi_data = roi_gray.ravel()
        data=np.vstack((data,roi_data))



data = data[1: , : ]
print(data.shape)



###############################################################################
# geting database ready 

n_samples = data.shape[0]
X = data
n_features = data.shape[1]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
#X_train, X_test, y_train, y_test = train_test_split(
   # X, y, test_size=0.25, random_state=42)

X_test = X.astype(float)



###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

# Load the pca classifier
pca = joblib.load("pca_clf.pkl")

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
x_test_pca = pca.transform(X_test)

print("done in %0.3fs" % (time() - t0))


target_names = np.load('target.npy')
# Load the classifier
clf = joblib.load("recognition_clf.pkl")


print("Predicting people's names on the test set")
print(target_names)
y_pred = clf.predict(x_test_pca)
for i,each in enumerate(y_pred):
    print(each)
    print(target_names[np.int(each[0])])
    roi = X[i].reshape((150,150))
    roi=np.array(roi, dtype=np.uint8)
    plt.imshow(roi, cmap = 'gray', interpolation = 'bicubic')
    plt.show()








