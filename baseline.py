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
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/file175"  # Replace with the actual path to your folder
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
     if len(trace_files1)-10==102:
       for trace_file in trace_files1:
          if trace_file!='7007.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
             seismic_tracenn = (seismic_tracenn - np.min(seismic_tracenn)) / (np.max(seismic_tracenn) - np.min(seismic_tracenn))
             p1.append(seismic_tracenn)
          if trace_file=='7007.npy':
             trace_path = os.path.join(trace_p, trace_file)
             seismic_output = np.load(trace_path)
             p2.append(seismic_output)
          
          
       #lpf fiter           
       # Convert the list of vectors to a NumPy array
       vectors_as_array1 = np.array(p1)
       # Merged vectors into a single vector
       merged_vector1 = vectors_as_array1.flatten()
       window_size=10
       # Apply the moving average filter or low pass filter
       filtered_signal_1=merged_vector1-np.convolve(merged_vector1 , np.ones(window_size)/window_size, mode='same')
       mm.append(filtered_signal_1)
X=np.array(mm)
y=np.array(np.array(p2))
print(X.shape)
#split into train test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_reshaped = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

# Create the LSTM model
model = Sequential([
    LSTM(units=32, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=y_train_reshaped.shape[1])  # Output units match the size of y vectors
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train_reshaped, y_train_reshaped, epochs=10, batch_size=2, validation_data=(X_test_reshaped, y_test_reshaped))
