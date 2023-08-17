import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
from scipy.interpolate import interp2d
import pywt
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/traces_data/56783_56709/56709"  # Replace with the actual path to your folder

# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
print(trace_files)
p=[]
# Loop through the trace files and visualize each actual trace
for trace_file in trace_files:
      if trace_file!='op.npy':
         trace_path = os.path.join(folder_path, trace_file)
         seismic_trace = np.load(trace_path)
         p.append(seismic_trace)
#plotting the output trace        
for trace_file in trace_files:
    if trace_file=='op.npy':
       trace_path = os.path.join(folder_path, trace_file)
       seismic_trace = np.load(trace_path)
       time_axis = np.linspace(0, len(seismic_trace), len(seismic_trace))
       plt.figure(figsize=(10, 6))
       plt.plot(time_axis, seismic_trace, color='red', linewidth=0.5)
       plt.xlabel("Time (s)")
       plt.ylabel("Amplitude")
       plt.title("Seismic output Trace")
       
    
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
plt.title("Seismic input Trace")


#ploting the merged input trace using normalization and using median filter
seismic_trace = (merged_vector - np.min(merged_vector)) / (np.max(merged_vector) - np.min(merged_vector))
window_size = 5  # Adjust the window size as needed
denoised_trace1 = medfilt(seismic_trace, kernel_size=window_size)
time_axis = np.linspace(0, len(merged_vector), len(merged_vector))
plt.figure(figsize=(10, 6))
plt.plot(time_axis, denoised_trace1, color='black', linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Seismic processed input trace using medfilt")

#ploting the merged input trace using normalization and using lpf
seismic_trace = (merged_vector - np.min(merged_vector)) / (np.max(merged_vector) - np.min(merged_vector))
window_size=11
# Apply the moving average filter or low pass filter
filtered_signal = np.convolve(seismic_trace , np.ones(window_size)/window_size, mode='same')
time_axis = np.linspace(0, len(merged_vector), len(merged_vector))
plt.figure(figsize=(10, 6))
plt.plot(time_axis, filtered_signal, color='black', linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Seismic processed input trace using lpf")

#ploting the merged input trace using normalization and using weiner filter
seismic_trace = (merged_vector - np.min(merged_vector)) / (np.max(merged_vector) - np.min(merged_vector))
denoised_signal2 = wiener(seismic_trace)
time_axis = np.linspace(0, len(merged_vector), len(merged_vector))
plt.figure(figsize=(10, 6))
plt.plot(time_axis, denoised_signal2, color='black', linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Seismic processed input trace using weiner")
         
#ploting the merged input trace using normalization and using  gaussian filter
seismic_trace = (merged_vector - np.min(merged_vector)) / (np.max(merged_vector) - np.min(merged_vector))
sigma = 3.0  # Adjust the standard deviation as needed
denoised_trace3 = gaussian_filter(seismic_trace, sigma=sigma)
time_axis = np.linspace(0, len(merged_vector), len(merged_vector))
plt.figure(figsize=(10, 6))
plt.plot(time_axis, denoised_trace3, color='black', linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Seismic processed input trace using gaussian filter")

#ploting the merged input trace using normalization and using pywt filter  Wavelet Denoising
seismic_trace = (merged_vector - np.min(merged_vector)) / (np.max(merged_vector) - np.min(merged_vector))
wavelet = 'db4'  # Choose a wavelet function
level = 2        # Adjust the decomposition level as needed
coeffs = pywt.wavedec(seismic_trace, wavelet, level=level)
sigma=0.6745
value = sigma * np.median(np.abs(coeffs[-level]))
coeffs[1:] = [pywt.threshold(coef, value, mode='soft') for coef in coeffs[1:]]
denoised_trace4 = pywt.waverec(coeffs, wavelet)
time_axis = np.linspace(0, len(denoised_trace4), len(denoised_trace4))
plt.figure(figsize=(10, 6))
plt.plot(time_axis, denoised_trace4, color='black', linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Seismic processed input trace using pywt")

##coherent noise removal

#F-K (Frequency-Wavenumber) Filtering
seismic_trace = (merged_vector - np.min(merged_vector)) / (np.max(merged_vector) - np.min(merged_vector))
num_samples = len(seismic_trace)
num_traces = 1  # Since you are working with a single trace
seismic_trace_2d = seismic_trace.reshape(num_samples, num_traces)
def fk_filter(trace):
    fk_data = np.fft.fft2(trace)
    fk_data[abs(fk_data) < 0.1] = 0  # Adjust threshold as needed
    denoised_trace = np.fft.ifft2(fk_data).real
    return denoised_trace
fk_denoised_trace = fk_filter(seismic_trace_2d)
time_axis = np.linspace(0, len(fk_denoised_trace), len(fk_denoised_trace))
plt.figure(figsize=(10, 6))
plt.plot(time_axis, fk_denoised_trace, color='black', linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Seismic processed input trace using F-K (Frequency-Wavenumber) Filtering")

#combination median filter + F-K (Frequency-Wavenumber) Filtering

seismic_trace = (merged_vector - np.min(merged_vector)) / (np.max(merged_vector) - np.min(merged_vector))
window_size = 5  # Adjust the window size as needed
denoised_tracee = medfilt(seismic_trace, kernel_size=window_size)
num_samples = len(denoised_tracee)
num_traces = 1  # Since you are working with a single trace
seismic_trace_2d = denoised_tracee.reshape(num_samples, num_traces)
def fk_filter(trace):
    fk_data = np.fft.fft2(trace)
    fk_data[abs(fk_data) < 0.1] = 0  # Adjust threshold as needed
    denoised_trace = np.fft.ifft2(fk_data).real
    return denoised_trace
fk_denoised_trace = fk_filter(seismic_trace_2d)
time_axis = np.linspace(0, len(fk_denoised_trace), len(fk_denoised_trace))
plt.figure(figsize=(10, 6))
plt.plot(time_axis, fk_denoised_trace, color='black', linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Seismic processed input trace using median filter + F-K (Frequency-Wavenumber) Filtering")

#combination gaussian filter + F-K (Frequency-Wavenumber) Filtering

seismic_trace = (merged_vector - np.min(merged_vector)) / (np.max(merged_vector) - np.min(merged_vector))
sigma = 3.0  # Adjust the standard deviation as needed
denoised_tracee1 = gaussian_filter(seismic_trace, sigma=sigma)
num_samples = len(denoised_tracee1)
num_traces = 1  # Since you are working with a single trace
seismic_trace_2dd = denoised_tracee1.reshape(num_samples, num_traces)
def fk_filter(trace):
    fk_data = np.fft.fft2(trace)
    fk_data[abs(fk_data) < 0.1] = 0  # Adjust threshold as needed
    denoised_trace = np.fft.ifft2(fk_data).real
    return denoised_trace
fk_denoised_trace = fk_filter(seismic_trace_2dd)
time_axis = np.linspace(0, len(fk_denoised_trace), len(fk_denoised_trace))
plt.figure(figsize=(10, 6))
plt.plot(time_axis, fk_denoised_trace, color='black', linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Seismic processed input trace using gaussian filter + F-K (Frequency-Wavenumber) Filtering")


