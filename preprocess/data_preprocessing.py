import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
import pywt
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/traces_data/56783_56709/56709"  # Replace with the actual path to your folder

# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
print(trace_files)

# Loop through the trace files and visualize each actual trace
#Seismic input Trace

for trace_file in trace_files[4:5]:
          if trace_file!='op.npy':
             trace_path = os.path.join(folder_path, trace_file)
             seismic_tracenn = np.load(trace_path)
             time_axis = np.linspace(0, len(seismic_tracenn), len(seismic_tracenn))
             plt.figure(figsize=(10, 6))
             plt.plot(time_axis, seismic_tracenn, color='black', linewidth=0.5)
             plt.xlabel("Time (s)")
             plt.ylabel("Amplitude")
             plt.title("Seismic input Trace")

#Seismic processed input trace using medfilt
for trace_file in trace_files[4:5]:
          if trace_file!='op.npy':
             trace_path = os.path.join(folder_path, trace_file)
             seismic_tracenn = np.load(trace_path)
             window_size = 51  # Adjust this value as needed
             flattened_trace = seismic_tracenn - medfilt(seismic_tracenn, kernel_size=window_size)
             time_axis = np.linspace(0, len(flattened_trace), len(flattened_trace))
             plt.figure(figsize=(10, 6))
             plt.plot(time_axis, flattened_trace, color='black', linewidth=0.5)
             plt.xlabel("Time (s)")
             plt.ylabel("Amplitude")
             plt.title("Seismic input Trace with medfilt")

#Seismic processed input trace using lpf        
for trace in trace_files[13:14]:
 if trace!='op.npy':
    trace_path = os.path.join(folder_path, trace)
    #loading the data
    seismic_trace = np.load(trace_path)
    #normalizing the data
    seismic_trace = (seismic_trace - np.min(seismic_trace)) / (np.max(seismic_trace) - np.min(seismic_trace))
    window_size=10
    # Apply the moving average filter or low pass filter
    filtered_signal=seismic_trace-np.convolve(seismic_trace , np.ones(window_size)/window_size, mode='same')
    time_axis = np.linspace(0, len(seismic_trace), len(seismic_trace))
    plt.figure(figsize=(10, 6))    
    plt.plot(time_axis, filtered_signal, 'black')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title("Seismic processed input trace using lpf")
    plt.show()

#Seismic processed input trace using wiener
for trace in trace_files[4:5]:  
      if trace!='op.npy': 
        m=os.path.join(folder_path,trace)
        seismic_trace = np.load(m)
        window_size = 51  # Adjust this value as needed
        flattened_trace = seismic_trace - wiener(seismic_trace)
        time_axis = np.linspace(0, len(flattened_trace), len(flattened_trace))
        plt.figure(figsize=(10, 6))    
        plt.plot(time_axis, flattened_trace, 'black')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title("Seismic processed input trace using wiener")
        plt.show()

#Seismic processed input trace using Gaussianfilter
for trace in trace_files[4:5]:  
      if trace!='op.npy': 
        m=os.path.join(folder_path,trace)
        seismic_trace = np.load(m)
        sigma = 6.0  # Adjust the standard deviation as needed
        
        flattened_trace = seismic_trace - gaussian_filter(seismic_trace, sigma=sigma)
        time_axis = np.linspace(0, len(flattened_trace), len(flattened_trace))
        plt.figure(figsize=(10, 6))    
        plt.plot(time_axis, flattened_trace, 'black')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title("Seismic processed input trace using gaussian filter")
        plt.show()
        
#pywt filter  Wavelet Denoising
for trace in trace_files[4:5]:  
      if trace!='op.npy': 
        m=os.path.join(folder_path,trace)
        seismic_trace = np.load(m)
        wavelet = 'db4'  # Choose a wavelet function
        level = 2        # Adjust the decomposition level as needed
        sigma=0.6745
        coeffs = pywt.wavedec(seismic_trace, wavelet, level=level)
        value = sigma * np.median(np.abs(coeffs[-level]))
        coeffs[1:] = [pywt.threshold(coef, value, mode='soft') for coef in coeffs[1:]]
        
        flattened_trace = seismic_trace - pywt.waverec(coeffs, wavelet)[:8001]
        time_axis = np.linspace(0, len(flattened_trace), len(flattened_trace))
        plt.figure(figsize=(10, 6))    
        plt.plot(time_axis, flattened_trace, 'black')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title("Seismic processed input trace using wavelet denoising filter")
        plt.show()
        
