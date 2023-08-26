import numpy as np
import cv2
import os
from scipy.signal import convolve
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.signal import medfilt
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.signal import correlate
from scipy.signal import medfilt
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.stats import mode
import pywt
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/Andaman"  # Replace with the actual path to your folder
mm=[]
# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
print(trace_files)
p2=[]
s=[]
"""
for j in trace_files:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     trace_files1 = [f for f in os.listdir(trace_p)]
     subfolder_path = os.path.join(trace_p, 'filtered_merged')
     if not os.path.exists(subfolder_path):
            os.mkdir(subfolder_path)
     
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
     resampled_array1 = np.interp(np.linspace(0, 1, 8050),
                                     np.linspace(0, 1, 8001),
                                     vectors_as_array1)
     
     plot_filename = f"{j}.npy" 
     path1=  os.path.join(trace_p,"filtered_merged")
     output_file= os.path.join(path1, plot_filename)
     np.save(output_file, resampled_array1)
""" 
"""   
#s2 (reducing lag)
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/Andaman"  # Replace with the actual path to your folder
mm=[]
# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
print(trace_files)
p2=[]
r=[]
s=[]
for j in trace_files:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     trace_files1 = [f for f in os.listdir(trace_p)]
     subfolder_path = os.path.join(trace_p, 'filtered_merged_reducing_lag')
     if not os.path.exists(subfolder_path):
            os.mkdir(subfolder_path)
     
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
     resampled_array1 = np.interp(np.linspace(0, 1, 8050),
                                     np.linspace(0, 1, 8001),
                                     vectors_as_array1)
     r.append(resampled_array1)
XX=np.array(r) 
y=np.array(p2)  
for k,i, j in zip(trace_files,XX, y):
    correlation = correlate(i, j)
    lag = np.argmax(correlation) - (len(i) - 1)
    aligned_input_trace = np.roll(i, -lag)
    plot_filename = f"{k}.npy" 
    path1=  os.path.join('/media/mleng/HDD/projects2023/Surabhi_seismic/data/Andaman',k)
    path2=  os.path.join(path1,"filtered_merged_reducing_lag")
    output_file= os.path.join(path2, plot_filename)
    np.save(output_file,aligned_input_trace)
"""
"""
for j in trace_files:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     trace_files1 = [f for f in os.listdir(trace_p)]
     subfolder_path = os.path.join(trace_p, 'kurtosis_variance_ratio')
     if not os.path.exists(subfolder_path):
            os.mkdir(subfolder_path)
     
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
     resampled_array1 = np.interp(np.linspace(0, 1, 8050),
                                     np.linspace(0, 1, 8001),
                                     vectors_as_array1)
     window_size=500
     mean_values = np.convolve(resampled_array1 , np.ones(window_size) / window_size, mode='same')
     variance_values = np.convolve((resampled_array1 - mean_values) ** 2, np.ones(window_size) / window_size, mode='same')
     std_deviation_values = np.sqrt(variance_values)
     local_skewness = np.convolve((resampled_array1 - mean_values) ** 3, np.ones(window_size) / window_size, mode='same') / std_deviation_values ** 3
     local_kurtosis = np.convolve((resampled_array1 - mean_values) ** 4, np.ones(window_size) / window_size, mode='same') / std_deviation_values ** 4
     skewness_to_std_ratio= local_skewness / std_deviation_values
     mean_std_ratio = mean_values / std_deviation_values
     skewness_kurtosis_ratio = local_skewness / local_kurtosis
     variance_mean_ratio= variance_values/ mean_values
     kurtosis_variance_ratio = local_kurtosis / variance_values
     plot_filename = f"{j}.npy" 
     path1=  os.path.join(trace_p,"kurtosis_variance_ratio")
     output_file= os.path.join(path1, plot_filename)
     np.save(output_file,kurtosis_variance_ratio)

"""
#lag features
p2=[]
r=[]
s=[]
for j in trace_files:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     trace_files1 = [f for f in os.listdir(trace_p)]
     subfolder_path = os.path.join(trace_p, 'lag_removal_kurtosis_variance_ratio')
     if not os.path.exists(subfolder_path):
            os.mkdir(subfolder_path)
     
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
     resampled_array1 = np.interp(np.linspace(0, 1, 8050),
                                     np.linspace(0, 1, 8001),
                                     vectors_as_array1)
     r.append(resampled_array1)
XX=np.array(r) 
y=np.array(p2)  
for k,i, j in zip(trace_files,XX, y):
    correlation = correlate(i, j)
    lag = np.argmax(correlation) - (len(i) - 1)
    aligned_input_trace = np.roll(i, -lag)
    window_size=500
    mean_values = np.convolve(aligned_input_trace , np.ones(window_size) / window_size, mode='same')
    variance_values = np.convolve((aligned_input_trace - mean_values) ** 2, np.ones(window_size) / window_size, mode='same')
    std_deviation_values = np.sqrt(variance_values)
    local_skewness = np.convolve((aligned_input_trace - mean_values) ** 3, np.ones(window_size) / window_size, mode='same') / std_deviation_values ** 3
    local_kurtosis = np.convolve((aligned_input_trace - mean_values) ** 4, np.ones(window_size) / window_size, mode='same') / std_deviation_values ** 4
    skewness_to_std_ratio= local_skewness / std_deviation_values
    mean_std_ratio = mean_values / std_deviation_values
    skewness_kurtosis_ratio = local_skewness / local_kurtosis
    variance_mean_ratio= variance_values/ mean_values
    kurtosis_variance_ratio = local_kurtosis / variance_values
    plot_filename = f"{k}.npy" 
    path1=  os.path.join('/media/mleng/HDD/projects2023/Surabhi_seismic/data/Andaman',k)
    path2=  os.path.join(path1,"lag_removal_kurtosis_variance_ratio")
    output_file= os.path.join(path2, plot_filename)
    np.save(output_file,kurtosis_variance_ratio)