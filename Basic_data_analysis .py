import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.signal import butter, lfilter

#Extracting the data from a folder to find the correlation of features within a folder
data_folder = '/media/mleng/HDD/projects2023/Surabhi_seismic/traces_data/56783_56709/56757'
folders = os.listdir(data_folder)
#for loop is formed all the 101 input data is stored in a list input_data and output in list output_data
input_data=[]
output_data=[]
for subfolders in folders:
    
    if subfolders!="op.npy":
        input_path = os.path.join(data_folder, subfolders)
        input_array = np.load(input_path)
        input_data.append(input_array)   
    if subfolders=="op.npy":
           output_path = os.path.join(data_folder, subfolders)
           output_array = np.load(output_path)
           output_data.append(output_array)  
m=[]
#finding the correlation between the independent  features
for j in input_data:
    for k in input_data:
       m.append(np.corrcoef(j, k)[0,1])
#print(m)
m1=[]
#finding the correlation between the independent feature and dependent feature
for j in input_data:
    for k in output_data:
        m1.append(np.corrcoef(k[:8001],j)[0,1])
#print(m1)

## Data preprocessing (using normalization and different filters)
# Path to a folder containing the trace files and analysing the effect on normalization and lowpass filter on sesmic input trace.
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/traces_data/56783_56709/56709"  # Replace with the actual path to your folder

# List all files in the folder
trace_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]


# Loop through the trace files and visualize each actual trace
for trace_file in trace_files[10:11]:
 if trace_file!='op.npy':
    trace_path = os.path.join(folder_path, trace_file)
    seismic_trace = np.load(trace_path)
    time_axis = np.linspace(0, len(seismic_trace), len(seismic_trace))
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, seismic_trace, color='black', linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Seismic input Trace")

# Loop through the trace files and visualize each processesed trace using low pass filter
for trace_file in trace_files[10:11]:
 if trace_file!='op.npy':
    trace_path = os.path.join(folder_path, trace_file)
    #loading the data
    seismic_trace = np.load(trace_path)
    #normalizing the data
    seismic_trace = (seismic_trace - np.min(seismic_trace)) / (np.max(seismic_trace) - np.min(seismic_trace))
    window_size=50
    # Apply the moving average filter or low pass filter
    filtered_signal = np.convolve(seismic_trace , np.ones(window_size)/window_size, mode='same')
    time_axis = np.linspace(0, len(seismic_trace), len(seismic_trace))
    plt.figure(figsize=(10, 6))    
    plt.plot(time_axis, filtered_signal, 'black')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title("Seismic processed input trace using lpf")
    plt.show()
  
# Loop through the trace files and visualize each trace using weiner filter
for trace_file in trace_files[10:11]:
 if trace_file!='op.npy':
    trace_path = os.path.join(folder_path, trace_file)
    seismic_trace = np.load(trace_path)
    #normalizing the data
    seismic_trace = (seismic_trace - np.min(seismic_trace)) / (np.max(seismic_trace) - np.min(seismic_trace))
    # Apply the Wiener filter to remove noise
    denoised_signal = wiener(seismic_trace)
    time_axis = np.linspace(0, len(seismic_trace), len(seismic_trace))
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_axis, denoised_signal, 'black')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title("Seismic processed input trace using weiner filter")
    plt.show()

# Loop through the trace files and visualize each trace using notch filter
for trace_file in trace_files[10:11]:
 if trace_file!='op.npy':
    trace_path = os.path.join(folder_path, trace_file)
    seismic_trace = np.load(trace_path)
    #normalizing the data
    seismic_trace = (seismic_trace - np.min(seismic_trace)) / (np.max(seismic_trace) - np.min(seismic_trace))
    fs = 500  #sample frequency
    powerline_freq = 50  # frequency of the powerline interference 
    
    def notch_filter(data, fs, powerline_freq, Q=30):
      nyquist = 0.5 * fs
      notch_freq = powerline_freq / nyquist
      b, a = butter(4, [notch_freq - 1.0 / Q, notch_freq + 1.0 / Q], btype='bandstop')
      filtered_data = lfilter(b, a, data)
      return filtered_data

    filtered_data = notch_filter(seismic_trace, fs, powerline_freq)
    time_axis = np.linspace(0, len(seismic_trace), len(seismic_trace))
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, denoised_signal, 'black')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title("Seismic processed input trace using notch filter")
    plt.show()
