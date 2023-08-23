import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pywt
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/data1/56783_56709"  # Replace with the actual path to your folder
mm=[]
# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
print(len(trace_files))
p2=[]
for j in trace_files:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     trace_files1 = [f for f in os.listdir(trace_p)]
        
     p1=[]
     for trace_file in trace_files1:
          if trace_file!='op.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
             #seismic_tracenn = (seismic_tracenn - np.min(seismic_tracenn)) / (np.max(seismic_tracenn) - np.min(seismic_tracenn))
             window_size=5
             # Apply the moving average filter or low pass filter
             filtered_trace=seismic_tracenn-np.convolve(seismic_tracenn, np.ones(window_size)/window_size, mode='same')
             p1.append(filtered_trace)
          if trace_file=='op.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_output = np.load(trace_path)
             p2.append(seismic_output)
          
     sum_vector = [sum(elements) for elements in zip(*p1)]        
     # Convert the list of vectors to a NumPy array
     vectors_as_array1 = np.array(sum_vector)
     mm.append(vectors_as_array1)
X=np.array(mm)
y=np.array(np.array(p2))
print("before sampling X",X.shape)
print("y",y.shape)
#Resampling the datapoints such that X andd y will have same number of datapoints
new_length = max(X.shape[1], y.shape[1])
# Initialize arrays to store resampled data
resampled_array1 = np.zeros((X.shape[0], new_length))
# Resample each row of the input arrays in X
for i in range(X.shape[0]):
    resampled_array1[i] = np.interp(np.linspace(0, 1, new_length),
                                    np.linspace(0, 1, X.shape[1]),
                                    X[i])
XX=resampled_array1
print("After sampling X",XX.shape)
print("y",y.shape)  
for i,j in zip(XX,y):
   print('correlation between X and y',np.corrcoef(i,j)[0,1])
   
# Feature Engineering

# window size
window_size = 500

# Initialize arrays to store the results
mean_values = np.empty_like(XX)
variance_values = np.empty_like(XX)
std_deviation_values = np.empty_like(XX)
local_skewness = np.empty((XX.shape[0], XX.shape[1]))  
local_kurtosis = np.empty((XX.shape[0], XX.shape[1]))

# Calculate mean, variance, and standard deviation within the sliding window for each row
for i in range(XX.shape[0]):
    mean_values[i] = np.convolve(XX[i], np.ones(window_size) / window_size, mode='same')
    variance_values[i] = np.convolve((XX[i] - mean_values[i]) ** 2, np.ones(window_size) / window_size, mode='same')
    std_deviation_values[i] = np.sqrt(variance_values[i])
    local_skewness[i] = np.convolve((XX[i] - mean_values[i]) ** 3, np.ones(window_size) / window_size, mode='same') / std_deviation_values[i] ** 3
    local_kurtosis[i] = np.convolve((XX[i] - mean_values[i]) ** 4, np.ones(window_size) / window_size, mode='same') / std_deviation_values[i] ** 4


for i,j in zip(mean_values,y):
   print('correlation2 between X and y',np.corrcoef(i,j)[0,1])
print('*'*100)
for i,j in zip(variance_values,y):
   print('correlation3 between X and y',np.corrcoef(i,j)[0,1])

print('*'*100)
for i,j in zip(std_deviation_values,y):
   print('correlation4 between X and y',np.corrcoef(i,j)[0,1])
   

print('*'*100)
for i,j in zip(local_skewness,y):
   print('correlation5 between X and y',np.corrcoef(i,j)[0,1])
   
print('*'*100)
for i,j in zip(local_kurtosis,y):
   print('correlation6 between X and y',np.corrcoef(i,j)[0,1])
# Define shift sizes
shift_sizes = np.arange(1, 21)
# Initialize arrays to store positive and negative shifted features
shifted_data_positive = np.empty((XX.shape[0], XX.shape[1], len(shift_sizes)))  # Only positive shifts
shifted_data_negative = np.empty((XX.shape[0], XX.shape[1], len(shift_sizes)))  # Only negative shifts

# Create positive and negative shifted features for data
for i in range(XX.shape[0]):
    for j, shift_size in enumerate(shift_sizes):
        shifted_data_positive[i, :, j] = np.concatenate((XX[i, shift_size:], np.full(shift_size, -3)))
        shifted_data_negative[i, :, j] = np.concatenate((np.full(shift_size, -3), XX[i, :-shift_size]))
shifted_data_positive_reshaped = shifted_data_positive.reshape(-1, XX.shape[1])
shifted_data_negative_reshaped = shifted_data_negative.reshape(-1, XX.shape[1])
print(shifted_data_positive.shape)
print('*'*100)
for i,j in zip(shifted_data_positive_reshaped,y):
   print('correlation7 between X and y',np.corrcoef(i,j)[0,1])

for i,j in zip(shifted_data_negative_reshaped,y):
   print('correlation8 between X and y',np.corrcoef(i,j)[0,1])

num_traces, trace_length = XX.shape
feature_matrix = np.zeros((num_traces, 0))
       