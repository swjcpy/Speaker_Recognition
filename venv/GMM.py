import MFCC
import librosa
import numpy as np
import os
from sklearn.mixture import GMM
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

train_X = np.load("trainingData.npy")
test_X = np.load("testingData.npy")
train_y = np.load("trainingLabel.npy")
test_y = np.load("testingLabel.npy")

X_train = np.reshape(train_X, (train_X.shape[0],28*28))
X_test = np.reshape(test_X, (test_X.shape[0],28*28))
y_train = np.zeros(train_y.shape[0])
y_test = np.zeros(test_y.shape[0])
for i in range(train_y.shape[0]):
    for j in range(11):
        if train_y[i,j] == 1:
            y_train[i] = j
            break
        j += 1
for i in range(test_y.shape[0]):
    for j in range(11):
        if test_y[i,j] == 1:
            y_test[i] = j
            break
        j += 1

# Try GMMs using different types of covariances.
n_classes = 11
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=50))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print('Train accuracy: %.1f' % train_accuracy)

    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    print('Test accuracy: %.1f' % test_accuracy)
