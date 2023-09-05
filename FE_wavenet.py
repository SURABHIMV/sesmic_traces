
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (TimeDistributed, Dropout, BatchNormalization, Flatten, Convolution1D, Activation, Input, Dense, LSTM, Lambda, Bidirectional,
                                     Add, AveragePooling1D, Multiply, GRU, GRUCell, LSTMCell, SimpleRNNCell, SimpleRNN, TimeDistributed, RNN,
                                     RepeatVector, Conv1D, MaxPooling1D, Concatenate, GlobalAveragePooling1D, UpSampling1D)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers
import pywt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv1D, Activation, Add, Multiply
from tensorflow.keras.models import Model
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/Andaman"  # Replace with the actual path to your folder
m=[]

# List all files in the folder
trace_files = [f for f in os.listdir(folder_path)]
p2=[]
s=[]
c=0
for j in trace_files[:100]:
     trace_p = os.path.join(folder_path, j)
     # List all files in the folder
     c=c+1
     trace_files1 = [f for f in os.listdir(trace_p)]

     p1=[]
     for trace_file in trace_files1:

          if trace_file!='op.npy' and trace_file.endswith('.npy'):
             trace_path = os.path.join(trace_p, trace_file)
             seismic_tracenn = np.load(trace_path)
             seismic_tracenn=seismic_tracenn/1.25
             #seismic_tracenn = (seismic_tracenn - np.min(seismic_tracenn)) / (np.max(seismic_tracenn) - np.min(seismic_tracenn))
             window_size=5
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
     time_axis1 = np.linspace(0, len(resampled_array1), len(resampled_array1))
     plt.plot(time_axis1, resampled_array1, color='black', linewidth=0.5)
     plt.xlabel("Time (s)")
     plt.ylabel("Amplitude")
     plt.title("Seismic batch output Trace")
     plt.show()
     for i in resampled_array1 :
        m.append(i)


print(set(p2))
print(len(p2))
print(len(m))
print(len(s))
#Based on the given data a dataframe is created
df=pd.DataFrame(data={'cdp':s,'input_filtered_merged':m,'output':p2})
df=df[:500000]
print(df)

#Adding new features to dataframe
window_size=2000
shift_size = 1000
df['backward'] = df['input_filtered_merged'].shift(periods=-shift_size, fill_value=0)
df['mean_input_signal'] = df['backward'].rolling(window=window_size, min_periods=1).mean()
#variance
df['variance_input_signal'] = df['backward'].rolling(window=window_size, min_periods=1).var()
df['stddev_input_signal'] = df['backward'].rolling(window=window_size, min_periods=1).std()
df['rolling_skewness'] = df['backward'].rolling(window=window_size, min_periods=1).apply(lambda x: pd.Series(x).skew())
df['rolling_kurtosis'] = df['backward'].rolling(window=window_size, min_periods=1).apply(lambda x: pd.Series(x).kurtosis())
df['skewness_to_std_ratio'] = df['rolling_skewness'] / df['stddev_input_signal']
df['mean_std_ratio']=df['mean_input_signal']/df['stddev_input_signal']
df['variance_mean_ratio']=df['variance_input_signal']/df['mean_input_signal']
df['skewness_kurtosis_ratio']=df['rolling_skewness']/df['rolling_kurtosis']
df['quantile_input_signal_01'] = df['backward'].rolling(window=10, min_periods=1).quantile(0.01)
df['quantile_input_signal_05'] = df['backward'].rolling(window=10, min_periods=1).quantile(0.05)
df['quantile_input_signal_25'] = df['backward'].rolling(window=10, min_periods=1).quantile(0.25)
df['quantile_input_signal_50'] = df['backward'].rolling(window=10, min_periods=1).quantile(0.5)
df['quantile_input_signal_95'] = df['backward'].rolling(window=10, min_periods=1).quantile(0.95)
df['quantile_input_signal_75'] = df['backward'].rolling(window=10, min_periods=1).quantile(0.75)
df['quantile_input_signal_99'] = df['backward'].rolling(window=10, min_periods=1).quantile(0.99)
df['rolling_min'] = df['backward'].rolling(window=10, min_periods=1).min()
df['rolling_max'] = df['backward'].rolling(window=10, min_periods=1).max()

##FFT
#Real fft (np.real and np.imaginary extract  the real and imaginary part of fft)
fft_input_merged_values = fft(np.require(df['backward'], requirements=['ALIGNED']))
fft_input_merged_values = np.real(fft_input_merged_values)
# Convert the FFT result into a pandas Series
realFFT_inm = pd.Series(fft_input_merged_values)

df['rolling_R_input_merge_mean'] = realFFT_inm.rolling(window=window_size, min_periods=1).mean()
df['rolling_R_input_merge_std'] = realFFT_inm.rolling(window=window_size, min_periods=1).std()
df['rolling_R_input_merge_max'] = realFFT_inm.rolling(window=window_size, min_periods=1).max()
df['rolling_R_input_merge_min'] = realFFT_inm.rolling(window=window_size, min_periods=1).min()
df['rolling_R_input_merge_var'] = realFFT_inm.rolling(window=window_size, min_periods=1).var()
df['rolling_R_input_merge_median'] = realFFT_inm.rolling(window=window_size, min_periods=1).median()
fft_input_merged_values1 = np.imag(fft_input_merged_values)
# Convert the FFT result into a pandas Series
realFFT_inm1 = pd.Series(fft_input_merged_values1)

#df['rolling_I_input_merge_mean'] = realFFT_inm1.rolling(window=window_size, min_periods=1).mean()
#df['rolling_I_input_merge_std'] = realFFT_inm1.rolling(window=window_size, min_periods=1).std()
#df['rolling_I_input_merge_max'] = realFFT_inm1.rolling(window=window_size, min_periods=1).max()
#df['rolling_I_input_merge_min'] = realFFT_inm1.rolling(window=window_size, min_periods=1).min()
#df['rolling_I_input_merge_var'] = realFFT_inm1.rolling(window=window_size, min_periods=1).var()

#def rolling_hilbert_mean(x):
#    return np.abs(hilbert(x)).mean()

# Calculate the rolling mean and assign it to a new column
#df['rolling_Hilbert_mean'] = df['input_filtered_merged'].rolling(window=window_size, min_periods=1).apply(rolling_hilbert_mean)
#df['quantile_input_signal_99_abs'] = df['input_filtered_merged'].abs().rolling(window=window_size, min_periods=1).quantile(0.99)
#df['quantile_input_signal_.95_abs'] = df['input_filtered_merged'].abs().rolling(window=window_size, min_periods=1).quantile(0.95)
#df['quantile_input_signal_.75_abs'] = df['input_filtered_merged'].abs().rolling(window=window_size, min_periods=1).quantile(0.75)
#df['quantile_input_signal_.50_abs'] = df['input_filtered_merged'].abs().rolling(window=window_size, min_periods=1).quantile(0.50)
#df['quantile_input_signal_.25_abs'] = df['input_filtered_merged'].abs().rolling(window=window_size, min_periods=1).quantile(0.25)
#df['quantile_input_signal_.05_abs'] = df['input_filtered_merged'].abs().rolling(window=window_size, min_periods=1).quantile(0.05)
#df['quantile_input_signal_.01_abs'] = df['input_filtered_merged'].abs().rolling(window=window_size, min_periods=1).quantile(0.01)
df['rolling_median_input_signal'] = df['backward'].rolling(window=10, min_periods=1).median()

print(df.corr()['output'])
df = df.dropna()
#counting the number of points in each class
print(df['output'].value_counts())
X=df.drop(['output','cdp'], axis=1)
y=df['output']
#split into train test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, shuffle=False)
print('X_train',X_train.shape)

print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
#X_train=np.array(X_train)
#X_train =X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test=np.array(X_test)
X_test =X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print('X_test',X_test.shape)

print(y_test.unique())
print(y_train.unique())

#model
def Regressor(shape_):
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
    
    inp = Input(shape = (shape_))
    
    x = wave_block(inp, 16, 3, 8)
    #x = wave_block(inp, 16, 3, 8)
    x = wave_block(x, 32, 3, 8)
    x = wave_block(x, 64, 3, 4)
    x = wave_block(x, 128, 3,1)

    out = Dense(1, name = 'out')(x)
    
    model =Model(inputs = inp, outputs = out)
    

    return model

# Hyperparameters

sequence_length =1   
num_epochs = 5
batch_size = 8050  # Batch size for training
num_layers = 10
num_filters = 64
dilation_depth = 9

# Create and compile the regression model
shape_=(None,27)
model=Regressor(shape_)
model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')



batch_size = 8050
num_samples = len(X_train)
num_batches = num_samples // batch_size
num_epochs = 20
train_losses=[]
val_losses=[]
train_batch_input=[]
train_batch_output=[]
train_output_original=[]

ep=[]
b=[]
#model training using the custom loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    epoch_train_loss = 0  
    for batch in range(num_batches):
        b.append(batch)
        ep.append(epoch)
        batch_start = batch * batch_size
        batch_end = (batch + 1) * batch_size
        batch_inputs = X_train[batch_start:batch_end]
        batch_inputs = X_train[batch_start:batch_end]
        batch_inputs=np.array(batch_inputs)
        batch_inputs=batch_inputs.reshape(batch_inputs.shape[0], 1, batch_inputs.shape[1])
        train_batch_input.append(batch_inputs)
        batch_targets = y_train[batch_start:batch_end]
        train_output_original.append(batch_targets)
        loss = model.train_on_batch(batch_inputs, batch_targets)
        epoch_train_loss += loss
       
        # Get predicted outputs for the current batch
        batch_outputs = model.predict(batch_inputs)
        train_batch_output.append(batch_outputs)

    epoch_train_loss /= num_batches  # Calculate average train loss for the epoch
    print(f"Average Train Loss: {epoch_train_loss:.4f}")
    
    train_losses.append(epoch_train_loss)  # Store train loss
    
    # Evaluate the model on the test data to get the validation loss
    val_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    val_losses.append(val_loss)  # Store validation loss

print(train_batch_output)

for i in train_batch_output:
    print('*'*100)
    data = tf.squeeze(i, axis=2)
    print(data)
    print(data.shape)
    print('*'*100)
    
#change file name to store the data in different folder
folder_path = "/media/mleng/HDD/projects2023/Surabhi_seismic/data/EXP/new_exp4_8"

# List all input train data and output batch in the folders
trace_files = [f for f in os.listdir(folder_path)]

#storing the data in separate folder
for j in trace_files:
 """  
 if j=='Batch_input_train':
   for i,k,l in zip(b,train_batch_input,ep):
     for j1 in k.columns:
        subfolder_path = os.path.join(folder_path,j)
        sub=os.path.join(subfolder_path,j1)
        if not os.path.exists(sub):
               os.mkdir(sub)
        if l==48:
           s=k[j1]
           time_axis1 = np.linspace(0, len(s), len(s))
           plt.figure(figsize=(10, 6))
           plt.plot(time_axis1, s, color='black', linewidth=0.5)
           plt.xlabel("Time (s)")
           plt.ylabel("Amplitude")
           plt.title("Seismic batch input Trace")
           # Save the plot in the folder
           plot_filename = f"epoch{l}_batch{i}.png" 
           #path1=  os.path.join(folder_path,j)
           plot_path = os.path.join(sub,plot_filename)
           plt.savefig(plot_path)
           plt.close()
 
 """
 if j=='batch_pred_output_train':
   for i,k,l in zip(b,train_batch_output,ep):
         d = tf.squeeze(k, axis=2)
         time_axis1 = np.linspace(0, len(d), len(d))
         plt.figure(figsize=(10, 6))
         plt.plot(time_axis1, d, color='black', linewidth=0.5)
         plt.xlabel("Time (s)")
         plt.ylabel("Amplitude")
         plt.title("Seismic predicted output Trace")
         # Save the plot in the folder
         plot_filename = f"epoch{l}_batch{i}.png" 
         path1=  os.path.join(folder_path,j)
         plot_path = os.path.join(path1,plot_filename)
         plt.savefig(plot_path)
         plt.close()


 elif j=='batch_original_output_train':
      for i,k,l in zip(b,train_output_original,ep):
            
            time_axis1 = np.linspace(0, len(k), len(k))
            plt.figure(figsize=(10, 6))
            plt.plot(time_axis1, k, color='black', linewidth=0.5)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Seismic original output train Trace")
            # Save the plot in the folder
            plot_filename = f"epoch{l}_batch{i}.png" 
            path1=  os.path.join(folder_path,j)
            plot_path = os.path.join(path1,plot_filename)
            plt.savefig(plot_path)
            plt.close()

