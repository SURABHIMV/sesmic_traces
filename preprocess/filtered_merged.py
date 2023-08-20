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

#Seismic input Trace
for j in trace_files:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     p=[]
     trace_files1 = [f for f in os.listdir(trace_p)]
     
     #median fiter        
     p1=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
             seismic_tracenn = (seismic_tracenn - np.min(seismic_tracenn)) / (np.max(seismic_tracenn) - np.min(seismic_tracenn))
             window_size = 51  # Adjust this value as needed
             flattened_trace = seismic_tracenn - medfilt(seismic_tracenn, kernel_size=window_size)
             p1.append(flattened_trace)
          
              
     # Convert the list of vectors to a NumPy array
     vectors_as_array1 = np.array(p1)
     # Merged vectors into a single vector
     merged_vector1 = vectors_as_array1.flatten()
     #plotting the input trace
     time_axis1 = np.linspace(0, len(merged_vector1), len(merged_vector1))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis1, merged_vector1, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic filtered_merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}.png" 
     path1=  os.path.join(trace_p,"median_filter_merged")
     plot_path = os.path.join(path1, plot_filename)
     plt.savefig(plot_path)
     plt.close()
     
     #lpf
     p2=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_trace_1= np.load(trace_path)
             seismic_trace_1 = (seismic_trace_1 - np.min(seismic_trace_1)) / (np.max(seismic_trace_1) - np.min(seismic_trace_1))
             window_size=10
             # Apply the moving average filter or low pass filter
             filtered_signal_1=seismic_trace_1-np.convolve(seismic_trace_1 , np.ones(window_size)/window_size, mode='same')
             p2.append(filtered_signal_1)
          
              
     # Convert the list of vectors to a NumPy array
     vectors_as_array2 = np.array(p2)
     # Merged vectors into a single vector
     merged_vector2 = vectors_as_array2.flatten()
     #plotting the input trace
     time_axis2 = np.linspace(0, len(merged_vector2), len(merged_vector2))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis2, merged_vector2, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic filtered_merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}.png" 
     path1=  os.path.join(trace_p,"lpf_merged")
     plot_path = os.path.join(path1, plot_filename)
     
     plt.savefig(plot_path)
     plt.close()
     
     
     #Seismic processed input trace using Gaussianfilter
     p4=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_trace_4= np.load(trace_path)
             seismic_trace_4 = (seismic_trace_4 - np.min(seismic_trace_4)) / (np.max(seismic_trace_4) - np.min(seismic_trace_4))
             sigma = 3.0  # Adjust the standard deviation as needed
             filtered_signal_4 = seismic_trace_4-gaussian_filter(seismic_trace_4, sigma=sigma)
             p4.append(filtered_signal_4)
          
     # Convert the list of vectors to a NumPy array
     vectors_as_array4 = np.array(p4)
     # Merged vectors into a single vector
     merged_vector4 = vectors_as_array4.flatten()
     #plotting the input trace
     time_axis4 = np.linspace(0, len(merged_vector4), len(merged_vector4))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis4, merged_vector4, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic filtered_merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}.png" 
     path1=  os.path.join(trace_p,"gaussian_filter_merged")
     plot_path = os.path.join(path1, plot_filename)
     plt.savefig(plot_path)
     plt.close()
       

  
     
