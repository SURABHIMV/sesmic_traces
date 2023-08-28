import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings("ignore")
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
from scipy.signal import medfilt
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.stats import mode
from scipy.signal import hilbert
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pywt
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/Andaman"  # Replace with the actual path to your folder
m=[]

# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
p2=[]
s=[]
p1=[]
c=0
for j in trace_files[:100]:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     c=c+1
     trace_files1 = [f for f in os.listdir(trace_p)]

     
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
             for i in seismic_output:
                 p2.append(i)
                 s.append(j)
     #merging the p2 list
     sum_vector = [sum(elements) for elements in zip(*p1)]  
     resampled_array1 = np.interp(np.linspace(0, 1, 8050),
                                     np.linspace(0, 1, 8001),
                                     sum_vector)
     for i in resampled_array1 :
        m.append(i)


print(set(p2))
print(len(p2))
print(len(m))
print(len(s))

df=pd.DataFrame(data={'cdp':s,'input_filtered_merged':m,'output':p2})
df=df[:80000]
print(df)
window_size=500 
#mean
df['mean_input_signal'] = df['input_filtered_merged'].rolling(window=window_size, min_periods=1).mean()
#variance
df['variance_input_signal'] = df['input_filtered_merged'].rolling(window=window_size, min_periods=1).var()
df['stddev_input_signal'] = df['input_filtered_merged'].rolling(window=window_size, min_periods=1).std()
df['rolling_skewness'] = df['input_filtered_merged'].rolling(window=window_size, min_periods=1).apply(lambda x: pd.Series(x).skew())
df['rolling_kurtosis'] = df['input_filtered_merged'].rolling(window=window_size, min_periods=1).apply(lambda x: pd.Series(x).kurtosis())
df['skewness_to_std_ratio2'] = df['rolling_skewness'] / df['stddev_input_signal']
df['variance_mean_ratio2']=df['variance_input_signal']/df['mean_input_signal']
#checking the correlation of independent variables with that of dependent variable
print(df.corr()['output'])
df = df.dropna()
#counting the number of points in each class
print(df['output'].value_counts())
X=df.drop(['output','cdp'], axis=1)
y=df['output']
#split into train test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)
print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
print(y_test.unique())
print(y_train.unique())


#model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(8,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_absolute_error')
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
history1=model.fit(X_train, y_train, batch_size=8050, epochs=50, validation_data=(X_test, y_test), callbacks=[reduce_lr_callback])
# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

#plotting the train and test loss
plt.figure(figsize=(12, 6))



# Plot training & validation loss values

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()




