# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
from sklearn import model_selection, metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
import scipy

# %%---------------------------------------------------------------------------
#           Load in Data
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'orientation_and_gravity.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing

#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 5

n_samples = len(data)-1
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

class_names = ["crunches", "planks", "bicycle crunches", "mountain climbers"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1] 
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1]) #label index

    
X = np.asarray(X) #feature vectors
Y = np.asarray(Y)
n_features = len(X)

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

precisions = []
recalls = []
accuracies = []
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    
    tree = tree.fit(X_train,Y[train_index])
    y_pred = tree.predict(X_test)
    conf = confusion_matrix(Y[test_index], y_pred) 

    for i in range(len(conf)):
        true_pos = conf[i][i]
        total = 0
        for j in range(len(conf)):
            total += conf[j][i]
        precision = true_pos/total
        precisions.append(precision)

    for i in range(len(conf)):
        true_pos = conf[i][i]
        total = 0
        for j in range(len(conf)):
            total += conf[i][j]
        recall = true_pos/total
        recalls.append(recall)
    accuracies.append(accuracy_score(Y[test_index],y_pred))

print("average accuracy:", np.mean(accuracies))
print("average precision:", np.mean(precisions))
print("average recall:", np.mean(recalls))

tree = tree.fit(X,Y)

export_graphviz(tree, out_file='tree.dot', feature_names = feature_names, class_names = class_names)

with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)

