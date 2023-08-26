import numpy as np
import cv2
import os
from scipy.stats import entropy
from scipy.signal import convolve
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.signal import medfilt
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.stats import mode
from scipy.signal import hilbert
import pywt
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/Andaman"  # Replace with the actual path to your folder
mm=[]
# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
print(trace_files)
p2=[]
s=[]
for j in trace_files[:100]:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     trace_files1 = [f for f in os.listdir(trace_p)]

     p1=[]
     for trace_file in trace_files1:
          if trace_file!='op.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
             seismic_tracenn=seismic_tracenn/1.25
             #seismic_tracenn = (seismic_tracenn - np.min(seismic_tracenn)) / (np.max(seismic_tracenn) - np.min(seismic_tracenn))
             window_size=100
             # Apply the moving average filter or low pass filter
             filtered_trace=seismic_tracenn-np.convolve(seismic_tracenn, np.ones(window_size)/window_size, mode='same')
             p1.append(filtered_trace)
            
          if trace_file=='op.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_output = np.load(trace_path)
             p2.append(seismic_output)

     #merging the p2 list
     sum_vector = [sum(elements) for elements in zip(*p1)]        
     # Convert the list of vectors to a NumPy array
     vectors_as_array1 = np.array(sum_vector)
     mm.append(vectors_as_array1)
X=np.array(mm[:-1])
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
#statistical features
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

#1
for i,j in zip(mean_values,y):
   print('correlation_mean between X and y',np.corrcoef(i,j)[0,1])
for i in mean_values:
              #plotting the input trace
              time_axis1 = np.linspace(0, len(i), len(i))
              plt.figure(figsize=(10, 6))
              plt.plot(time_axis1, i, color='black', linewidth=0.5)
              plt.xlabel("Time (s)")
              plt.ylabel("Amplitude")
              plt.title("Seismic filtered_merged Trace")
print('*'*60)
#2
for i,j in zip(variance_values,y):
   print('correlation_variance between X and y',np.corrcoef(i,j)[0,1])
#3
print('*'*60)
for i,j in zip(std_deviation_values,y):
   print('correlation_std between X and y',np.corrcoef(i,j)[0,1])
#4
print('*'*60)
for i,j in zip(local_skewness,y):
   print('correlation_skewness between X and y',np.corrcoef(i,j)[0,1])
#5   
print('*'*60)
for i,j in zip(local_kurtosis,y):
   print('correlation_kurtosis between X and y',np.corrcoef(i,j)[0,1])
   
# Initialize arrays to store the results
min_values = np.empty_like(XX)
max_values = np.empty_like(XX)

# Calculate rolling minimum and maximum using scipy's functions for each row
for i in range(XX.shape[0]):
    min_values[i] = minimum_filter1d(XX[i], size=window_size, mode='wrap')
    max_values[i] = maximum_filter1d(XX[i], size=window_size, mode='wrap')
#6
print('*'*60)
for i,j in zip(min_values,y):
   print('correlation_min between X and y',np.corrcoef(i,j)[0,1]) 
#7   
print('*'*60)
for i,j in zip(max_values,y):
   print('correlation_max between X and y',np.corrcoef(i,j)[0,1])                       
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

print(shifted_data_positive)



#8
print('*'*60)
for i,j in zip(shifted_data_positive_reshaped,y):
   print('correlation_shift_positive between X and y',np.corrcoef(i,j)[0,1])

#9
print('*'*60)
for i,j in zip(shifted_data_negative_reshaped,y):
   print('correlation_shift_negative between X and y',np.corrcoef(i,j)[0,1])

#Median
median_values = np.empty_like(XX)
  
window_size=501
# Calculate rolling median within the sliding window for each row
for i in range(XX.shape[0]):
    median_values[i] = medfilt(XX[i], kernel_size=window_size)
#10
print('*'*60)
for i,j in zip(median_values,y):
   print('correlation_median between X and y',np.corrcoef(i,j)[0,1]) 
"""
for i in median_values:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace")
"""    
#percentile

percentile1_values = np.empty_like(XX)
percentile5_values = np.empty_like(XX)
percentile10_values = np.empty_like(XX)
percentile25_values = np.empty_like(XX)
percentile50_values = np.empty_like(XX)
percentile75_values = np.empty_like(XX)
percentile90_values = np.empty_like(XX)
percentile95_values = np.empty_like(XX)
percentile99_values = np.empty_like(XX)
for i in range(XX.shape[0]):
    percentile1_values[i] = np.percentile(XX[i], 1, axis=0, keepdims=True)
    percentile5_values[i] = np.percentile(XX[i], 5, axis=0, keepdims=True)
    percentile10_values[i] = np.percentile(XX[i], 10, axis=0, keepdims=True)
    percentile25_values[i] = np.percentile(XX[i], 25, axis=0, keepdims=True)
    percentile50_values[i] = np.percentile(XX[i], 50, axis=0, keepdims=True)
    percentile75_values[i] = np.percentile(XX[i], 75, axis=0, keepdims=True)
    percentile90_values[i] = np.percentile(XX[i], 90, axis=0, keepdims=True)
    percentile95_values[i] = np.percentile(XX[i], 95, axis=0, keepdims=True)
    percentile99_values[i] = np.percentile(XX[i], 99, axis=0, keepdims=True)
#11
print('*'*60)
for i,j in zip(percentile1_values,y):
   print('correlation_percentile1_values  X and y',np.corrcoef(i,j)[0,1])

#12
print('*'*60)
for i,j in zip(percentile5_values,y):
   print('correlation_percentile5_values  X and y',np.corrcoef(i,j)[0,1])
#13
print('*'*60)
for i,j in zip(percentile10_values,y):
   print('correlation_percentile10_values  X and y',np.corrcoef(i,j)[0,1])
#14
print('*'*60)
for i,j in zip(percentile25_values,y):
   print('correlation_percentile25_values  X and y',np.corrcoef(i,j)[0,1])

#15
print('*'*60)
for i,j in zip(percentile50_values,y):
   print('correlation_percentile50_values  X and y',np.corrcoef(i,j)[0,1])

#16
print('*'*60)
for i,j in zip(percentile75_values,y):
   print('correlation_percentile75_values  X and y',np.corrcoef(i,j)[0,1])

#17
print('*'*60)
for i,j in zip(percentile90_values,y):
   print('correlation_percentile90_values  X and y',np.corrcoef(i,j)[0,1])

#18
print('*'*60)
for i,j in zip(percentile95_values,y):
   print('correlation_percentile95_values  X and y',np.corrcoef(i,j)[0,1])
   
#19
print('*'*60)
for i,j in zip(percentile99_values,y):
   print('correlation_percentile99_values  X and y',np.corrcoef(i,j)[0,1])

#mode 

mode_values = np.empty_like(XX)

# Calculate rolling median and mode within the sliding window for each row
for i in range(XX.shape[0]):
    mode_values[i], _ = mode(XX[i], axis=0)
#20
print('*'*60)
for i,j in zip(mode_values,y):
   print('correlation_percentile_mode_values  X and y',np.corrcoef(i,j)[0,1])
"""
for i in mode_values:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace")
"""
#21

# Initialize array to store RMS values
rms_values = np.empty_like(XX)

# Calculate RMS within the sliding window for each row
for i in range(XX.shape[0]):
    # Calculate the squared values
    squared_values = XX[i] ** 2
    
    # Calculate the moving average of squared values
    moving_average_squared = np.convolve(squared_values, np.ones(window_size) / window_size, mode='same')
    
    # Calculate RMS values as the square root of the moving average of squared values
    rms_values[i] = np.sqrt(moving_average_squared)

print('*'*60)
for i,j in zip(rms_values,y):
   print('correlation_rms_values  X and y',np.corrcoef(i,j)[0,1])

#Backwardshift
#23
shift_size = 1000
shifted_vector = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector[i] = np.concatenate((XX[i,shift_size:], np.zeros(shift_size)))
"""
for i in shifted_vector:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace") 
"""
print('*'*60)
for i,j in zip(shifted_vector,y):
   print('correlation_Backwardshift_1000  X and y',np.corrcoef(i,j)[0,1])
 
#22
shift_size = 1100
shifted_vector1 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector1[i] = np.concatenate((XX[i,shift_size:], np.zeros(shift_size)))
"""
for i in shifted_vector1:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace")  
"""
print('*'*60)
for i,j in zip(shifted_vector1,y):
   print('correlation_Backwardshift_1100  X and y',np.corrcoef(i,j)[0,1])
   
#23
shift_size = 1200
shifted_vector2 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector2[i] = np.concatenate((XX[i,shift_size:], np.zeros(shift_size)))
"""
for i in shifted_vector2:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace") 
"""
print('*'*60)
for i,j in zip(shifted_vector2,y):
   print('correlation_Backwardshift_1200  X and y',np.corrcoef(i,j)[0,1])

#24
shift_size = 1300
shifted_vector3 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector3[i] = np.concatenate((XX[i,shift_size:], np.zeros(shift_size)))
"""
for i in shifted_vector3:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace") 
"""
print('*'*60)
for i,j in zip(shifted_vector3,y):
   print('correlation_Backwardshift_1300  X and y',np.corrcoef(i,j)[0,1])
   
#25
shift_size = 1350
shifted_vector4 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector4[i] = np.concatenate((XX[i,shift_size:], np.zeros(shift_size)))
"""
for i in shifted_vector3:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace") 
"""
print('*'*60)
for i,j in zip(shifted_vector4,y):
   print('correlation_Backwardshift_1350  X and y',np.corrcoef(i,j)[0,1])
   
#26
shift_size = 1400
shifted_vector5 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector5[i] = np.concatenate((XX[i,shift_size:], np.zeros(shift_size)))
"""
for i in shifted_vector4:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace")
"""
print('*'*60)
for i,j in zip(shifted_vector5,y):
   print('correlation_Backwardshift_1400  X and y',np.corrcoef(i,j)[0,1])
   
#27
shift_size = 1500
shifted_vector6 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector6[i] = np.concatenate((XX[i,shift_size:], np.zeros(shift_size)))
"""
for i in shifted_vector5:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace")   
"""
print('*'*60)
for i,j in zip(shifted_vector6,y):
   print('correlation_Backwardshift_1500  X and y',np.corrcoef(i,j)[0,1])
   
#28
shift_size = 1500
shifted_vector7 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector7[i] = np.concatenate((XX[i,shift_size:], np.zeros(shift_size)))

for i in shifted_vector7:
           """
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace") """  

print('*'*60)
for i,j in zip(shifted_vector7,y):
   print('correlation_Backwardshift_1500  X and y',np.corrcoef(i,j)[0,1])
   

#forwardshift
#29
shift_size = 100
shifted_vector8 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector8[i] = np.concatenate((np.zeros(shift_size),XX[i,:-shift_size]))
"""
for i in shifted_vector7:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace")

"""

print('*'*60)
for i,j in zip(shifted_vector8,y):
   print('correlation_forwardshift_100  X and y',np.corrcoef(i,j)[0,1])
   
#30
shift_size = 200
shifted_vector9 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector9[i] = np.concatenate((np.zeros(shift_size),XX[i,:-shift_size]))
"""
for i in shifted_vector8:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace")
"""


print('*'*60)
for i,j in zip(shifted_vector9,y):
   print('correlation_forwardshift_200  X and y',np.corrcoef(i,j)[0,1])

#shifted vector mean, variance, std,skewness,kurtosis
shift_size = 300
shifted_vector10 = np.empty_like(XX)
for i in range(XX.shape[0]):
   # Perform the backward shift and fill with zeros
   shifted_vector10[i] = np.concatenate((np.zeros(shift_size),XX[i,:-shift_size]))

# Initialize arrays to store the results
mean_values1 = np.empty_like(shifted_vector10)
variance_values1 = np.empty_like(shifted_vector10)
std_deviation_values1 = np.empty_like(shifted_vector10)
local_skewness1= np.empty((shifted_vector10.shape[0], shifted_vector10.shape[1]))  
local_kurtosis1 = np.empty((shifted_vector10.shape[0], shifted_vector10.shape[1]))

# Calculate mean, variance, and standard deviation within the sliding window for each row
for i in range(shifted_vector10.shape[0]):
    mean_values1[i] = np.convolve(shifted_vector10[i], np.ones(window_size) / window_size, mode='same')
    variance_values1[i] = np.convolve((shifted_vector10[i] - mean_values[i]) ** 2, np.ones(window_size) / window_size, mode='same')
    std_deviation_values1[i] = np.sqrt(variance_values[i])
    local_skewness1[i] = np.convolve((XX[i] - mean_values[i]) ** 3, np.ones(window_size) / window_size, mode='same') / std_deviation_values[i] ** 3
    local_kurtosis1[i] = np.convolve((XX[i] - mean_values[i]) ** 4, np.ones(window_size) / window_size, mode='same') / std_deviation_values[i] ** 4

#31
for i,j in zip(mean_values1,y):
   print('correlation_backwardshift_1300_mean between X and y',np.corrcoef(i,j)[0,1])
print('*'*60)
#32
for i,j in zip(variance_values1,y):
   print('correlation_backwardshift_1300_variance between X and y',np.corrcoef(i,j)[0,1])
#33
print('*'*60)
for i,j in zip(std_deviation_values1,y):
   print('correlation_backwardshift_1300_std between X and y',np.corrcoef(i,j)[0,1])
#34
print('*'*60)
for i,j in zip(local_skewness1,y):
   print('correlation_backwardshift_1300_skewness between X and y',np.corrcoef(i,j)[0,1])
#35   
print('*'*60)
for i,j in zip(local_kurtosis1,y):
   print('correlation__backwardshift_1300_kurtosis between X and y',np.corrcoef(i,j)[0,1])
   


#36
# Initialize array to store energy values
energy_values = np.empty_like(XX)

# Calculate energy values within the sliding window for each row
for i in range(XX.shape[0]):
    squared_values = XX[i] ** 2
    energy_values[i] = np.convolve(squared_values, np.ones(window_size), mode='same')
"""
for i in energy_values:
           #plotting the input trace
           time_axis1 = np.linspace(0, len(i), len(i))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, i, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic filtered_merged Trace")"""
print('*'*60)
for i,j in zip(energy_values,y):
   print('correlation_energy_values_between X and y',np.corrcoef(i,j)[0,1])

#37

# Initialize array to store energy values
energy_values1 = np.empty_like(XX)

# Calculate energy values within the sliding window for each row
for i in range(shifted_vector10.shape[0]):
    squared_values = shifted_vector10[i] ** 2
    energy_values1[i] = np.convolve(squared_values, np.ones(window_size), mode='same')

print('*'*60)
for i,j in zip(energy_values1,y):
   print('correlation_shifted_energy_values_between X and y',np.corrcoef(i,j)[0,1])
   
#38
# Initialize array to store entropy values
entropy_values = np.empty_like(XX)

# Calculate entropy for each row
for i in range(XX.shape[0]):
    normalized_values = (XX[i] - np.min(XX[i])) / (np.max(XX[i]) - np.min(XX[i]))  
    entropy_values[i] = entropy(normalized_values, base=2) 
print(entropy_values.shape)

print('*'*60)
for i,j in zip(entropy_values,y):
   print('correlation_entropy_between X and y',np.corrcoef(i,j)[0,1])
   

#Frequency domain features

# Initialize arrays to store frequency domain features
dominant_frequencies = np.empty_like(XX)
spectral_entropy = np.empty_like(XX)
spectral_energy_distribution = np.empty((XX.shape[0], XX.shape[1]))

# Calculate frequency domain features for each trace
for i in range(XX.shape[0]):
    trace = XX[i]
    # Apply Fourier Transform
    fft_values = fft(trace)
    # Calculate magnitude spectrum
    magnitude_spectrum = np.abs(fft_values)
    # Calculate dominant frequency as the frequency with maximum magnitude
    dominant_frequency_index = np.argmax(magnitude_spectrum)
    dominant_frequencies[i] = dominant_frequency_index / len(trace)
    # Calculate spectral entropy of the magnitude spectrum
    normalized_spectrum = magnitude_spectrum / np.sum(magnitude_spectrum)
    spectral_entropy[i] = entropy(normalized_spectrum, base=2)
    # Calculate spectral energy distribution
    spectral_energy_distribution[i] = magnitude_spectrum


#39
print('*'*60)
for i,j in zip(dominant_frequencies,y):
   print('correlation_dominant_between X and y',np.corrcoef(i,j)[0,1])

#40   
print('*'*60)
for i,j in zip(spectral_entropy,y):
   print('correlation_spectral_entropy_between X and y',np.corrcoef(i,j)[0,1])

#41   
print('*'*60)
for i,j in zip(spectral_energy_distribution,y):
   print('correlation_spectral_energy_distribution_between X and y',np.corrcoef(i,j)[0,1])

# Initialize arrays to store frequency domain features
dominant_frequencies1 = np.empty_like(shifted_vector10)
spectral_entropy1 = np.empty_like(shifted_vector10)
spectral_energy_distribution1 = np.empty((shifted_vector10.shape[0], XX.shape[1]))

# Calculate frequency domain features for each trace
for i in range(shifted_vector10.shape[0]):
    trace = shifted_vector10[i]
    # Apply Fourier Transform
    fft_values = fft(trace)
    # Calculate magnitude spectrum
    magnitude_spectrum1 = np.abs(fft_values)
    # Calculate dominant frequency as the frequency with maximum magnitude
    dominant_frequency_index1 = np.argmax(magnitude_spectrum1)
    dominant_frequencies1[i] = dominant_frequency_index1 / len(trace)
    # Calculate spectral entropy of the magnitude spectrum
    normalized_spectrum1 = magnitude_spectrum1 / np.sum(magnitude_spectrum1)
    spectral_entropy1[i] = entropy(normalized_spectrum1, base=2)
    # Calculate spectral energy distribution
    spectral_energy_distribution1[i] = magnitude_spectrum1

#42
print('*'*60)
for i,j in zip(dominant_frequencies1,y):
   print('correlation_dominant_frequencies_shift_between X and y',np.corrcoef(i,j)[0,1])

#43  
print('*'*60)
for i,j in zip(spectral_entropy1,y):
   print('correlation_spectral_entropyshift_between X and y',np.corrcoef(i,j)[0,1])

#44   
print('*'*60)
for i,j in zip(spectral_energy_distribution1,y):
   print('correlation_shift_entropy_between X and y',np.corrcoef(i,j)[0,1])


#cross corelation (removing the lag between the input merged and ouput)
print('*'*60)
aligned=[]
from scipy.signal import correlate

for i, j in zip(XX, y):
    correlation = correlate(i, j)
    lag = np.argmax(correlation) - (len(i) - 1)
    aligned_input_trace = np.roll(i, -lag)
    aligned.append(aligned_input_trace)
    """
    time = np.linspace(0, len(i), len(i))
    plt.plot(time, aligned_input_trace, linewidth=0.5)
    plt.show()
    plt.plot(time, i, color='black', linewidth=0.5)
    plt.show()
    plt.plot(time, j, color='red', linewidth=0.5)
    plt.show()
   """
XX1= np.array(aligned) 
print(XX1.shape)
window_size = 500

# Initialize arrays to store the results
mean_values2 = np.empty_like(XX1)
variance_values2 = np.empty_like(XX1)
std_deviation_values2 = np.empty_like(XX1)
local_skewness2 = np.empty((XX1.shape[0], XX1.shape[1]))  
local_kurtosis2 = np.empty((XX1.shape[0], XX1.shape[1]))
local_kurtosis2 = np.empty((XX1.shape[0], XX1.shape[1]))
skewness_to_std_ratio2=np.empty((XX1.shape[0], XX1.shape[1]))
mean_std_ratio2 = np.empty((XX1.shape[0], XX1.shape[1]))
skewness_kurtosis_ratio2 = np.empty((XX1.shape[0], XX1.shape[1]))
variance_mean_ratio2=np.empty((XX1.shape[0], XX1.shape[1]))
kurtosis_variance_ratio2=np.empty((XX1.shape[0], XX1.shape[1]))
# Calculate mean, variance, and standard deviation within the sliding window for each row
for i in range(XX1.shape[0]):
    mean_values2[i] = np.convolve(XX1[i], np.ones(window_size) / window_size, mode='same')
    variance_values2[i] = np.convolve((XX1[i] - mean_values[i]) ** 2, np.ones(window_size) / window_size, mode='same')
    std_deviation_values2[i] = np.sqrt(variance_values2[i])
    local_skewness2[i] = np.convolve((XX1[i] - mean_values[i]) ** 3, np.ones(window_size) / window_size, mode='same') / std_deviation_values2[i] ** 3
    local_kurtosis2[i] = np.convolve((XX1[i] - mean_values[i]) ** 4, np.ones(window_size) / window_size, mode='same') / std_deviation_values2[i] ** 4
    skewness_to_std_ratio2[i] = local_skewness2[i] / std_deviation_values2[i]
    mean_std_ratio2[i] = mean_values2[i] / std_deviation_values2[i]
    skewness_kurtosis_ratio2[i] = local_skewness2[i] / local_kurtosis2[i]
    variance_mean_ratio2[i]= variance_values2[i]/ mean_values2[i]
    kurtosis_variance_ratio2[i] = local_kurtosis2[i] / variance_values2[i]
#45
for i,j in zip(mean_values2,y):
   print('correlation_mean between X and y',np.corrcoef(i,j)[0,1])
print('*'*60)
#46
for i,j in zip(variance_values2,y):
   print('correlation_variance between X and y',np.corrcoef(i,j)[0,1])
#47
print('*'*60)
for i,j in zip(std_deviation_values2,y):
   print('correlation_std between X and y',np.corrcoef(i,j)[0,1])
#48
print('*'*60)
for i,j in zip(local_skewness2,y):
   print('correlation_skewness between X and y',np.corrcoef(i,j)[0,1])
#49
print('*'*60)
for i,j in zip(skewness_to_std_ratio2,y):
   print('correlation_skewness_to_std_ratio between X and y',np.corrcoef(i,j)[0,1])
#50
print('*'*60)
for i,j in zip(mean_std_ratio2,y):
   print('correlation_mean_std_ratio between X and y',np.corrcoef(i,j)[0,1])
#51
print('*'*60)
for i,j in zip(skewness_kurtosis_ratio2,y):
   print('correlation_skewness_kurtosis_ratio2 between X and y',np.corrcoef(i,j)[0,1])
   
#51
print('*'*60)
for i,j in zip(variance_mean_ratio2,y):
   print('correlation_variance_mean_ratio2 between X and y',np.corrcoef(i,j)[0,1])
   
#51
print('*'*60)
for i,j in zip(kurtosis_variance_ratio2,y):
   print('correlation_kurtosis_variance_ratio2 between X and y',np.corrcoef(i,j)[0,1])
   

#percentile (removing the lag between the input merged and ouput)

percentile1_values1 = np.empty_like(XX1)
percentile5_values1 = np.empty_like(XX1)
percentile10_values1 = np.empty_like(XX1)
percentile25_values1 = np.empty_like(XX1)
percentile50_values1 = np.empty_like(XX1)
percentile75_values1 = np.empty_like(XX1)
percentile90_values1 = np.empty_like(XX1)
percentile95_values1 = np.empty_like(XX1)
percentile99_values1 = np.empty_like(XX1)
for i in range(XX.shape[0]):
    percentile1_values1[i] = np.percentile(XX1[i], 1, axis=0, keepdims=True)
    percentile5_values1[i] = np.percentile(XX1[i], 5, axis=0, keepdims=True)
    percentile10_values1[i] = np.percentile(XX1[i], 10, axis=0, keepdims=True)
    percentile25_values1[i] = np.percentile(XX1[i], 25, axis=0, keepdims=True)
    percentile50_values1[i] = np.percentile(XX1[i], 50, axis=0, keepdims=True)
    percentile75_values1[i] = np.percentile(XX1[i], 75, axis=0, keepdims=True)
    percentile90_values1[i] = np.percentile(XX1[i], 90, axis=0, keepdims=True)
    percentile95_values1[i] = np.percentile(XX1[i], 95, axis=0, keepdims=True)
    percentile99_values1[i] = np.percentile(XX1[i], 99, axis=0, keepdims=True)
#52
print('*'*60)
for i,j in zip(percentile1_values1,y):
   print('correlation_percentile1_values  X and y',np.corrcoef(i,j)[0,1])
"""
for i in percentile1_values1:
              #plotting the input trace
              time_axis1 = np.linspace(0, len(i), len(i))
              plt.figure(figsize=(10, 6))
              plt.plot(time_axis1, i, color='black', linewidth=0.5)
              plt.xlabel("Time (s)")
              plt.ylabel("Amplitude")
              plt.title("Seismic filtered_merged Trace")
"""
#53
print('*'*60)
for i,j in zip(percentile5_values1,y):
   print('correlation_percentile5_values  X and y',np.corrcoef(i,j)[0,1])
#54
print('*'*60)
for i,j in zip(percentile10_values1,y):
   print('correlation_percentile10_values  X and y',np.corrcoef(i,j)[0,1])
#55
print('*'*60)
for i,j in zip(percentile25_values1,y):
   print('correlation_percentile25_values  X and y',np.corrcoef(i,j)[0,1])

#56
print('*'*60)
for i,j in zip(percentile50_values1,y):
   print('correlation_percentile50_values  X and y',np.corrcoef(i,j)[0,1])

#57
print('*'*60)
for i,j in zip(percentile75_values1,y):
   print('correlation_percentile75_values  X and y',np.corrcoef(i,j)[0,1])

#58
print('*'*60)
for i,j in zip(percentile90_values1,y):
   print('correlation_percentile90_values  X and y',np.corrcoef(i,j)[0,1])

#59
print('*'*60)
for i,j in zip(percentile95_values1,y):
   print('correlation_percentile95_values  X and y',np.corrcoef(i,j)[0,1])
   
#60
print('*'*60)
for i,j in zip(percentile99_values1,y):
   print('correlation_percentile99_values  X and y',np.corrcoef(i,j)[0,1])
 
#61
       
envelope_values = np.empty_like(XX)

# Apply Hilbert transform to compute envelope for each trace
for i in range(XX.shape[0]):
    analytic_signal = hilbert(XX[i])
    envelope_values[i] = np.abs(analytic_signal)
print(envelope_values.shape)
print('*'*60)
for i,j in zip(envelope_values,y):
   print('correlation_envelope_values  X and y',np.corrcoef(i,j)[0,1])
"""
for i in envelope_values:
              #plotting the input trace
              time_axis1 = np.linspace(0, len(i), len(i))
              plt.figure(figsize=(10, 6))
              plt.plot(time_axis1, i, color='black', linewidth=0.5)
              plt.xlabel("Time (s)")
              plt.ylabel("Amplitude")
              plt.title("Seismic filtered_merged Trace")"""
        
#62
envelope_values1 = np.empty_like(XX1)

# Apply Hilbert transform to compute envelope for each trace
for i in range(XX1.shape[0]):
    analytic_signal1 = hilbert(XX1[i])
    envelope_values1[i] = np.abs(analytic_signal1)
print(envelope_values1.shape)
print('*'*60)
for i,j in zip(envelope_values1,y):
   print('correlation_envelope_values  X and y',np.corrcoef(i,j)[0,1])
for i in envelope_values1:
              #plotting the input trace
              time_axis1 = np.linspace(0, len(i), len(i))
              plt.figure(figsize=(10, 6))
              plt.plot(time_axis1, i, color='black', linewidth=0.5)
              plt.xlabel("Time (s)")
              plt.ylabel("Amplitude")
              plt.title("Seismic filtered_merged Trace")

#

envelope_values1 = np.empty_like(XX1)
peak_amplitude1=np.empty_like(XX1)
mean_amplitude1=np.empty_like(XX1)
std_deviation1=np.empty_like(XX1)
skewness1=np.empty_like(XX1)
kurt1=np.empty_like(XX1)
# Apply Hilbert transform to compute envelope for each trace
for i in range(XX1.shape[0]):
    analytic_signal1 = hilbert(XX1[i])
    envelope_values1[i] = np.abs(analytic_signal1)
    peak_amplitude1[i] = np.max(envelope_values1[i])
    mean_amplitude1[i] = np.mean(envelope_values1[i])
    std_deviation1[i] = np.std(envelope_values1[i])
    skewness1[i] = skew(envelope_values1[i])
    kurt1[i] = kurtosis(envelope_values[i])
    
print(envelope_values1.shape)
print('*'*60)
for i,j in zip(peak_amplitude1,y):
   print('correlation_envelope_values  X and y',np.corrcoef(i,j)[0,1])
for i in mean_amplitude1:
              #plotting the input trace
              time_axis1 = np.linspace(0, len(i), len(i))
              plt.figure(figsize=(10, 6))
              plt.plot(time_axis1, i, color='black', linewidth=0.5)
              plt.xlabel("Time (s)")
              plt.ylabel("Amplitude")
              plt.title("Seismic filtered_merged Trace")

