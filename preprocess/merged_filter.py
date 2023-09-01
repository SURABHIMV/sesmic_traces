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

for j in trace_files:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     p=[]
     trace_files1 = [f for f in os.listdir(trace_p)]
     
          
     p1=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
             seismic_tracenn = (seismic_tracenn - np.min(seismic_tracenn)) / (np.max(seismic_tracenn) - np.min(seismic_tracenn))
             p1.append(seismic_tracenn)
          
     #median fiter           
     sum_vector1 = [sum(elements) for elements in zip(*p1)] 
     window_size = 51  # Adjust this value as needed
     flattened_trace = sum_vector1 - medfilt(sum_vector1, kernel_size=window_size)
     #plotting the input trace
     time_axis1 = np.linspace(0, len(flattened_trace), len(flattened_trace))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis1, flattened_trace, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic merged_filter Trace")
     # Save the plot in the folder
     plot_filename = f"{j}.png" 
     path1=  os.path.join(trace_p,"merged_median_filter")
     plot_path = os.path.join(path1, plot_filename)
     plt.savefig(plot_path)
     plt.close()
     
     #lpf
     
     window_size=10
     # Apply the moving average filter or low pass filter
     filtered_signal_1=sum_vector1-np.convolve(sum_vector1 , np.ones(window_size)/window_size, mode='same')
     #plotting the input trace
     time_axis2 = np.linspace(0, len(filtered_signal_1), len(filtered_signal_1))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis2, filtered_signal_1, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic merged_filter Trace")
     # Save the plot in the folder
     plot_filename = f"{j}.png" 
     path1=  os.path.join(trace_p,"merged_lpf")
     plot_path = os.path.join(path1, plot_filename)
     plt.savefig(plot_path)
     plt.close()
     
     
     #Seismic processed input trace using Gaussianfilter
     
     sigma = 3.0  # Adjust the standard deviation as needed
     filtered_signal_4 = sum_vector1-gaussian_filter(sum_vector1, sigma=sigma)
     #plotting the input trace
     time_axis4 = np.linspace(0, len(filtered_signal_4), len(filtered_signal_4))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis4, filtered_signal_4, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic merged_filter Trace")
     # Save the plot in the folder
     plot_filename = f"{j}.png" 
     path1=  os.path.join(trace_p,"merged_gaussian_filter")
     plot_path = os.path.join(path1, plot_filename)
     plt.savefig(plot_path)
     plt.close()
       

   
     







