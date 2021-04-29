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

#print(data[0])
# %%---------------------------------------------------------------------------
#
#		                    Pre-processing

#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
#reoriented = np.asarray([reorient(data[i,4], data[i,3], data[i,2]) for i in range(len(data))])
#print(reoriented[0])
#reoriented_data_with_timestamps = np.append(data[:,0:1],data,axis=1)
#print(reoriented_data_with_timestamps[0])
#data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

#print(data[0])
# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

# TODO: list the class labels that you collected data for in the order of label_index (defined while collecting data)
class_names = ["crunches", "planks", "bicycle crunches", "mountain climbers"] #...

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1] 
    #print(window)
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


# TODO: split data into train and test datasets using 10-fold cross validation

cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
precisions = []
recalls = []
accuracies = []
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
for train_index, test_index in cv.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    
    tree = tree.fit(X_train,Y[train_index])
    y_pred = tree.predict(X_test) #goal is y_pred = Y[test_index]
    conf = confusion_matrix(Y[test_index], y_pred) 
    #print(conf)

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


# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
print("average accuracy:", np.mean(accuracies))
print("average precision:", np.mean(precisions))
print("average recall:", np.mean(recalls))


# TODO: train the decision tree classifier on entire dataset

tree = tree.fit(X,Y)


# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line

export_graphviz(tree, out_file='tree.dot', feature_names = feature_names, class_names = class_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)

