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
     for trace_file in trace_files1:
          if trace_file!='7007.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
             p.append(seismic_tracenn)
          elif trace_file=='7007.npy':
              trace_path = os.path.join(trace_p, trace_file)
              seismic_trace_o = np.load(trace_path)
              time_axis = np.linspace(0, len(seismic_trace_o), len(seismic_trace_o))
              plt.figure(figsize=(10, 6))
              plt.plot(time_axis, seismic_trace_o, color='red', linewidth=0.5)
              plt.xlabel("Time (s)")
              plt.ylabel("Amplitude")
              plt.title("Seismic output Trace")
              # Save the plot in the folder
              plot_filename = f"{j}_plot.png" 
              plot_path = os.path.join("/media/mleng/HDD/projects2023/Surabhi_seismic/data/expirement_merged_out", plot_filename)
              plt.savefig(plot_path)
              plt.close()
              
     # Convert the list of vectors to a NumPy array
     vectors_as_array = np.array(p)
     # Merged vectors into a single vector
     merged_vector = vectors_as_array.flatten()
     #plotting the input trace
     time_axis = np.linspace(0, len(merged_vector), len(merged_vector))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis, merged_vector, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic input merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}_plot.png" 
     plot_path = os.path.join("/media/mleng/HDD/projects2023/Surabhi_seismic/data/expirement_merged_input_trace", plot_filename)
     plt.savefig(plot_path)
     plt.close()
     
     #median fiter        
     p1=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
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
     plt.title("Seismic input merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}_plot.png" 
     plot_path = os.path.join("/media/mleng/HDD/projects2023/Surabhi_seismic/data/expirement_merged_median_filter_", plot_filename)
     plt.savefig(plot_path)
     plt.close()
     
     #lpf
     p2=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_trace_1= np.load(trace_path)
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
     plt.title("Seismic input merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}_plot.png" 
     plot_path = os.path.join("/media/mleng/HDD/projects2023/Surabhi_seismic/data/expirement_merged_lpf", plot_filename)
     plt.savefig(plot_path)
     plt.close()
     
     
     #wiener filter
     p3=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_trace_2= np.load(trace_path)
             window_size = 51  # Adjust this value as needed
             filtered_signal_2 = seismic_trace_2 - wiener(seismic_trace_2)
             p3.append(filtered_signal_2)
          
              
     # Convert the list of vectors to a NumPy array
     vectors_as_array3 = np.array(p3)
     # Merged vectors into a single vector
     merged_vector3 = vectors_as_array3.flatten()
     #plotting the input trace
     time_axis3 = np.linspace(0, len(merged_vector3), len(merged_vector3))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis3, merged_vector3, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic input merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}_plot.png" 
     plot_path = os.path.join("/media/mleng/HDD/projects2023/Surabhi_seismic/data/expirement_merged_wiener_filter", plot_filename)
     plt.savefig(plot_path)
     plt.close()
     
     
     #Seismic processed input trace using Gaussianfilter
     p4=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_trace_4= np.load(trace_path)
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
     plt.title("Seismic input merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}_plot.png" 
     plot_path = os.path.join("/media/mleng/HDD/projects2023/Surabhi_seismic/data/expirement_merged_gaussian_filter", plot_filename)
     plt.savefig(plot_path)
     plt.close()
       

  
     #pywt filter  Wavelet Denoising
     p5=[]
     for trace_file in trace_files1:
          if trace_file!='7007.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_trace_5= np.load(trace_path)
             wavelet = 'db4'  # Choose a wavelet function
             level = 2        # Adjust the decomposition level as needed
             sigma=0.6745
             coeffs = pywt.wavedec(seismic_trace_4, wavelet, level=level)
             value = sigma * np.median(np.abs(coeffs[-level]))
             coeffs[1:] = [pywt.threshold(coef, value, mode='soft') for coef in coeffs[1:]]
             filtered_signal_5 = seismic_trace_5 - pywt.waverec(coeffs, wavelet)[:4501]
             p5.append(filtered_signal_5)
          
              
     # Convert the list of vectors to a NumPy array
     vectors_as_array5 = np.array(p5)
     # Merged vectors into a single vector
     merged_vector5 = vectors_as_array5.flatten()
     #plotting the input trace
     time_axis5 = np.linspace(0, len(merged_vector5), len(merged_vector5))
     plt.figure(figsize=(10, 6))
     plt.plot(time_axis5, merged_vector5, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic input merged Trace")
     # Save the plot in the folder
     plot_filename = f"{j}_plot.png" 
     plot_path = os.path.join("/media/mleng/HDD/projects2023/Surabhi_seismic/data/expirement_merged_wavelet_filter", plot_filename)
     plt.savefig(plot_path)
     plt.close()
     
   
     





