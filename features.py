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
from scipy.signal import find_peaks
import math 

#Compute Statistical Feature - Mean
def _compute_mean_features(window):
    return np.delete(np.mean(window[:,0:7], axis=0),[0,2,4,5])

#Compute Statistical Feature - Standard Deviation
def _compute_standard_deviation(window):
    return np.delete(np.std(window[:,0:7],axis = 0),[0,2,4,5])

def compute_orientation_magnitude(window):
    magnitude = []
    for i in range(len(window)): #qx, qz, qw, qy
        magnitude.append(math.sqrt(window[i,1]**2+ window[i,2]**2+ window[i,4]**2 + window[i,5]**2))
    return np.array(magnitude)

#Compute FFT feature - Dominant Frequency over Magnitude
def _compute_orientation_magnitude_dominant_frequency(window):
    magnitude = compute_orientation_magnitude(window)
    return [max(np.fft.rfft(magnitude).astype(float))]

#Compute entropy
def _entropy(window):
    hist, bin_edges = np.histogram(window, density=True) #change bin?
    hist = hist/hist.sum() 
    sum = 0
    for i in hist:
        if i!= 0:
            sum += i * math.log(i)
    return [sum] #needs to be 1D
    
def _compute_mean_orientation_amplitude(window):
    magnitude = compute_orientation_magnitude(window)
    threshold = np.mean(magnitude)
    peaks, _ = find_peaks(magnitude,height=threshold)
    mean_peak = np.mean(peaks)
    threshold = np.mean(-1*magnitude)
    troughs, _ = find_peaks(-1*magnitude, height=threshold)
    mean_trough = np.mean(troughs)
    mean_amplitude = mean_peak - mean_trough
    return [mean_amplitude]

def extract_features(window):

    x = []
    feature_names = []

    x.append(_compute_mean_features(window))
    x.append(_compute_standard_deviation(window))
    x.append(_compute_mean_orientation_amplitude(window))
    x.append(_compute_orientation_magnitude_dominant_frequency(window))
    x.append(_entropy(window))
    
    feature_names.append("qx_mean")
    feature_names.append("roll_mean")
    feature_names.append("pitch_mean")
    feature_names.append("qx_std")
    feature_names.append("roll_std")
    feature_names.append("pitch_std")
    feature_names.append("mean_orientation_amplitude")
    feature_names.append("orientation_dominant_frequency")
    feature_names.append("entropy")

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector