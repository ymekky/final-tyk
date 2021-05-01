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
    return np.delete(np.mean(window[:,0:7], axis=0),[0,2,4,5])

# TODO: define functions to compute more features
def _compute_standard_deviation(window):
    return np.delete(np.std(window[:,0:7],axis = 0),[0,2,4,5])

def compute_orientation_magnitude(window):
    magnitude = []
    for i in range(len(window)): #qx, qz, qw, qy
        magnitude.append(math.sqrt(window[i,1]**2+ window[i,2]**2+ window[i,4]**2 + window[i,5]**2))
    return np.array(magnitude)

def compute_gravity_magnitude(window):
    magnitude = []
    for i in range(len(window)):
        magnitude.append(math.sqrt(window[i,7]**2+ window[i,8]**2+ window[i,9]**2))
    return np.array(magnitude)

def _compute_orientation_magnitude_peaks(window):
    magnitude = compute_orientation_magnitude(window)
    peaks, _ = find_peaks(magnitude,prominence=0.15)
    return np.array([len(peaks)])

def _compute_gravity_magnitude_peaks(window):
    magnitude = compute_gravity_magnitude(window)
    peaks, _ = find_peaks(magnitude)
    return np.array([len(peaks)])

#Compute FFT features - Dominant Frequency over Magnitude
def _compute_orientation_magnitude_dominant_frequency(window):
    magnitude = compute_orientation_magnitude(window)
    return [max(np.fft.rfft(magnitude).astype(float))]

#Compute FFT features - Dominant Frequency over Magnitude
def _compute_gravity_magnitude_dominant_frequency(window):
    magnitude = compute_gravity_magnitude(window)
    return [max(np.fft.rfft(magnitude).astype(float))]

#Compute FFT features - Dominant Frequency over QX Axis
def _compute_qx_dominant_frequency(window):
    return [max(np.fft.rfft(window[:,1]).astype(float))]

#Compute FFT features - Dominant Frequency over Y Axis
def _compute_y_dominant_frequency(window):
    return np.fft.rfft(window[:2]).astype(float)

#Compute FFT features - Dominant Frequency over Z Axis (gravity)
def _compute_qz_dominant_frequency(window):
    return [max(np.fft.rfft(window[:,5]).astype(float))]

#Compute FFT features - Dominant Frequency over Z Axis
def _compute_w_dominant_frequency(window):
    return np.fft.rfft(window[:6]).astype(float)

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
    #print(mean_amplitude)
    return [mean_amplitude]
    


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
    #x.append(_compute_orientation_magnitude_peaks(window))
    #x.append(_compute_gravity_magnitude_peaks(window))
    #x.append(_compute_orientation_magnitude_dominant_frequency(window))
    #x.append(_compute_gravity_magnitude_dominant_frequency(window))
    #x.append(_entropy(window))
    #x.append(_compute_qx_dominant_frequency(window))
    x.append(_compute_mean_orientation_amplitude(window))
    #x.append(_compute_qz_dominant_frequency(window))
    
    #feature_names.append("yaw_mean")
    feature_names.append("qx_mean")
    #feature_names.append("qz_mean")
    feature_names.append("roll_mean")
    #feature_names.append("qw_mean")
    #feature_names.append("qy_mean")
    feature_names.append("pitch_mean")
    #feature_names.append("z_mean")
    #feature_names.append("y_mean")
    #feature_names.append("x_mean")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names
    
    #feature_names.append("yaw_std")
    feature_names.append("qx_std")
    #feature_names.append("qz_std")
    feature_names.append("roll_std")
    #feature_names.append("qw_std")
    #feature_names.append("qy_std")
    feature_names.append("pitch_std")
    #feature_names.append("z_std")
    #feature_names.append("y_std")
    #feature_names.append("x_std")
    
    #feature_names.append("orientantion_magnitude_peaks")
    #feature_names.append("gravity_magnitude_peaks")
    #feature_names.append("orientation_magnitude_dominant_frequency")
    #feature_names.append("gravity_magnitude_dominant_frequency")
    #feature_names.append("entropy")
    #feature_names.append("qx_dominant_frequency")
    feature_names.append("mean_orientation_amplitude")
    #feature_names.append("qz_dominant_frequency")

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector