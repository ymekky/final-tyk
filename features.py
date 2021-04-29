# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math 

def _compute_mean_features(window):
    return np.mean(window, axis=0)

# TODO: define functions to compute more features
def _compute_standard_deviation(window):
    return np.std(window,axis = 0)

def compute_magnitude(window):
    magnitude = []
    for i in range(len(window)):
        magnitude.append(math.sqrt(window[i,1]**2+ window[i,2]**2+ window[i,4]**2 + window[i,5]**2))
    return np.array(magnitude)

def _compute_magnitude_peaks(window):
    magnitude = compute_magnitude(window)
    peaks, _ = find_peaks(magnitude)
    return np.array([len(peaks)])


#Compute FFT features - Dominant Frequency over Magnitude
def _compute_magnitude_dominant_frequency(window):
    magnitude = compute_magnitude(window)
    return [max(np.fft.rfft(magnitude).astype(float))]

#Compute FFT features - Dominant Frequency over X Axis
def _compute_x_dominant_frequency(window):
    return np.fft.rfft(window[:1]).astype(float)

#Compute FFT features - Dominant Frequency over Y Axis
def _compute_y_dominant_frequency(window):
    return np.fft.rfft(window[:2]).astype(float)

#Compute FFT features - Dominant Frequency over Z Axis
def _compute_z_dominant_frequency(window):
    return np.fft.rfft(window[:5]).astype(float)

#Compute entropy
def _entropy(window):
    hist, bin_edges = np.histogram(window, density=True) #change bin?
    hist = hist/hist.sum() 
    sum = 0
    for i in hist:
        if i!= 0:
            sum += i * math.log(i)
    return [sum] #needs to be 1D
    
   


def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    
    x = []
    feature_names = []

    x.append(_compute_mean_features(window))
    x.append(_compute_standard_deviation(window))
    x.append(_compute_magnitude_peaks(window))
    x.append(_compute_magnitude_dominant_frequency(window))
    x.append(_entropy(window))
    feature_names.append("yaw_mean")
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("roll_mean")
    feature_names.append("qw_mean")
    feature_names.append("z_mean")
    feature_names.append("pitch_mean")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names
    feature_names.append("yaw_std")
    feature_names.append("x_std")
    feature_names.append("y_std")
    feature_names.append("roll_std")
    feature_names.append("qw_std")
    feature_names.append("z_std")
    feature_names.append("pitch_std")
    feature_names.append("magnitude_peaks")
    feature_names.append("magnitude_dominant_frequency")
    feature_names.append("entropy")
    
    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector