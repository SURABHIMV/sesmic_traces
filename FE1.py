import numpy as np
import cv2
import os
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
import pywt
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/Andaman"  # Replace with the actual path to your folder
m=[]

# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
p2=[]
s=[]
p1=[]
for j in trace_files[:100]:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
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
                 if i>=5000:
                     p2.append(1)
                     s.append(j)
                 else:
                     p2.append(0)
                     s.append(j)
     #merging the p2 list
     sum_vector = [sum(elements) for elements in zip(*p1)]  
     resampled_array1 = np.interp(np.linspace(0, 1, 8050),
                                     np.linspace(0, 1, 8001),
                                     sum_vector)
     for i in resampled_array1 :
        m.append(i)

print(set(p2))
df=pd.DataFrame(data={'cdp':s,'input_filtered_merged':m,'output':p2})
df=df[:100000]
print(df)
window_size=500 
#mean
df['mean_input_signal'] = df['input_filtered_merged'].rolling(window=window_size, min_periods=1).mean()
#variance
df['variance_input_signal'] = df['input_filtered_merged'].rolling(window=window_size, min_periods=1).var()
#checking the correlation of independent variables with that of dependent variable
print(df.corr()['output'])
df = df.dropna()
#counting the number of points in each class
print(df['output'].value_counts())
X=df.drop(['output','cdp'], axis=1)
y=df['output']
#split into train test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)
print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
print(y_test.unique())
print(y_train.unique())

#random forest
parameters = {'min_samples_split':[2,5,10],'max_depth':[1,5,10,20,50],'n_estimators':[10,100,500,1000,1200]}

r_cfl = RandomForestClassifier()
clf = RandomizedSearchCV(r_cfl, parameters, cv=10, scoring='roc_auc', n_jobs=4)
clf.fit(X_train, y_train)

# Get the best estimator from the search
best_rf_model = clf.best_estimator_

# Train the best model on the entire training data
best_rf_model.fit(X_train, y_train)

# Calibrate the best model
calibrated_rf = CalibratedClassifierCV(best_rf_model, method="sigmoid")
calibrated_rf.fit(X_train, y_train)

# Predict probabilities for training set
train_probs = calibrated_rf.predict_proba(X_train)
train_log_loss = log_loss(y_train, train_probs)
print("The train log loss is:", train_log_loss)

# Predict probabilities for test set
test_probs = calibrated_rf.predict_proba(X_test)
test_log_loss = log_loss(y_test, test_probs)
print("The test log loss is:", test_log_loss)

# Confusion matrix
C = confusion_matrix(y_test, calibrated_rf.predict(X_test))
print('Confusion Matrix:\n', C)

# Calculate and print recall (sensitivity)
recall = C[1, 1] / (C[1, 0] + C[1, 1])
print("Recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, calibrated_rf.predict(X_test))
print("F1 Score:", f1)





