import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
import pywt
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/file175"  # Replace with the actual path to your folder

# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
print(trace_files)

# Loop through the trace files and visualize each actual trace
#Seismic input Trace
for j in trace_files:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     trace_files1 = [f for f in os.listdir(trace_p)]
     #Seismic processed input trace using medfilt
     for trace_file in trace_files1:
          if trace_file!='7007.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
             seismic_tracenn = (seismic_tracenn - np.min(seismic_tracenn)) / (np.max(seismic_tracenn) - np.min(seismic_tracenn))
             window_size = 51  # Adjust this value as needed
             flattened_trace = seismic_tracenn - medfilt(seismic_tracenn, kernel_size=window_size)
             time_axis = np.linspace(0, len(flattened_trace), len(flattened_trace))
             plt.figure(figsize=(10, 6))
             plt.plot(time_axis, flattened_trace, color='black', linewidth=0.5)
             plt.xlabel("Time (s)")
             plt.ylabel("Amplitude")
             plt.title("Seismic input Trace with medfilt")
             # Save the plot in the folder
             plot_filename = f"{trace_file}.png" 
             #path1 = os.path.join("/media/mleng/HDD/projects2023/Surabhi_seismic/data/file175",j)
             path1=  os.path.join(trace_p,"individual_median_filter")
             plot_path = os.path.join(path1, plot_filename)
             plt.savefig(plot_path)
             plt.close()  

        #Seismic processed input trace using lpf        
     for trace in trace_files1:
          if trace!='7007.npy' and trace.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace)
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
             plt.xlabel('Time')
             plt.ylabel('Amplitude')
             plt.title("Seismic processed input trace using lpf")
             # Save the plot in the folder
             plot_filename = f"{trace}.png" 
             path1=  os.path.join(trace_p,"individual_lpf")
             plot_path = os.path.join(path1, plot_filename)
             plt.savefig(plot_path)
             plt.close()


     #Seismic processed input trace using Gaussianfilter
     for trace in trace_files1:  
       if trace!='7007.npy' and trace.endswith('.npy'): 
             m=os.path.join(trace_p,trace)
             seismic_trace = np.load(m)
             seismic_trace = (seismic_trace - np.min(seismic_trace)) / (np.max(seismic_trace) - np.min(seismic_trace))
             sigma = 6.0  # Adjust the standard deviation as needed
        
             flattened_trace = seismic_trace - gaussian_filter(seismic_trace, sigma=sigma)
             time_axis = np.linspace(0, len(flattened_trace), len(flattened_trace))
             plt.figure(figsize=(10, 6))    
             plt.plot(time_axis, flattened_trace, 'black')
             plt.xlabel('Time')
             plt.ylabel('Amplitude')
             plt.title("Seismic processed input trace using gaussian filter")
             # Save the plot in the folder
             plot_filename = f"{trace}.png" 
             path1=  os.path.join(trace_p,"individual_gaussian_filter")
             plot_path = os.path.join(path1, plot_filename)
             plt.savefig(plot_path)
             plt.close()
        
