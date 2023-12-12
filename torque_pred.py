from ngboost import NGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt

data_all_filepath = ".//data//data_all_K.mat"
torque_all_filepath = ".//data//torq_all_K.mat"
data_all = scipy.io.loadmat(data_all_filepath)
torque_all = scipy.io.loadmat(torque_all_filepath)

stride_filepaths = [".//data//input_ml_290.mat", 
".//data//input_ml_163.mat",
".//data//input_ml_161.mat",
".//data//input_ml_122.mat",
".//data//input_ml_67.mat"]
torque_filepaths = [".//data//output_ml_290.mat",
".//data//output_ml_163.mat",
".//data//output_ml_161.mat",
".//data//output_ml_122.mat",
".//data//output_ml_67.mat"]

# Butterworth filter for torque trajectory smoothing
# 3rd order with cutoff frequency of 100Hz
def low_pass(data, cutoff, fs, order=3):
    nyq = 0.5 * 1000
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

train_input_data = None
train_output_data = None
# Load the MATLAB file
for i in range(len(stride_filepaths)):
    input_strides = scipy.io.loadmat(stride_filepaths[i])
    output_torque = scipy.io.loadmat(torque_filepaths[i])
    stride_data = input_strides['result']
    torque_data = output_torque['yout']
    # Smooth torque data
    smooth_torque_data = low_pass(torque_data[:,0], cutoff=100, fs = 10000, order=3)
    if train_input_data is None:
        train_input_data = stride_data
        train_output_data = smooth_torque_data
    else:
        train_input_data = np.concatenate((train_input_data, stride_data), axis=0)
        train_output_data = np.concatenate((train_output_data, smooth_torque_data), axis=0)

# stride_test = train_input_data[:101,:]
# torque_test = train_output_data[:101]
# X = train_input_data[101:,:]
# Y = train_output_data[101:]
X = train_input_data
Y = train_output_data

#************** USE STROKE GAIT DATA *******************************###
stroke_input_data = None
stroke_output_data = None
# These are not the averaged 
stroke_stride_filepaths = [".//data//stroke_input_ml_290.mat", 
".//data//stroke_input_ml_163.mat",
".//data//stroke_input_ml_161.mat",
".//data//stroke_input_ml_122.mat",
".//data//stroke_input_ml_67.mat"]
stroke_torque_filepaths = [".//data//stroke_output_ml_290.mat",
".//data//stroke_output_ml_163.mat",
".//data//stroke_output_ml_161.mat",
".//data//stroke_output_ml_122.mat",
".//data//stroke_output_ml_67.mat"]
# Load the MATLAB file
# for i in range(len(stroke_stride_filepaths)):
#     input_strides = scipy.io.loadmat(stroke_stride_filepaths[i])
#     output_torque = scipy.io.loadmat(stroke_torque_filepaths[i])
#     stride_data = np.mean(input_strides['result'])
#     torque_data = np.mean(output_torque['yout'])
#     # Smooth torque data
#     smooth_torque_data = low_pass(torque_data[:,0], cutoff=100, fs = 10000, order=3)
#     if stroke_input_data is None:
#         stroke_input_data = stride_data
#         stroke_output_data = smooth_torque_data
#     else:
#         stroke_input_data = np.concatenate((stroke_input_data, stride_data), axis=0)
#         stroke_output_data = np.concatenate((stroke_output_data, smooth_torque_data), axis=0)

# This is averaged over all strides for each patient
stroke_data_all_filepath = ".//data//stroke_input_ml_s290.mat"
stroke_torque_all_filepath = ".//data//stroke_output_ml_s290.mat"
stroke_data_all = scipy.io.loadmat(stroke_data_all_filepath)
stroke_torque_all = scipy.io.loadmat(stroke_torque_all_filepath)

stroke_X_test = stroke_data_all['result']
stroke_Y_test = stroke_torque_all['yout']

#*******************************************************************##

################# Subject Able Body Average ############################################
# Access the variables stored in the MATLAB file
# Just level walking
# input_data = data_all['q_all'][:,:,:,0]  
# torque_data = torque_all['torq_all'][:,:,0] 

# subj1_X = input_data[:,:,0]  
# subj1_Y = torque_data[:,0]

# GRF, Ankle Angle, Shank Angle, Ankle Vel, Shank Vel, Phi, IMU_Vel, Ankle COP, Ankle Torque
# 11	1	6	8	10	5	9	20

# subj1X_test = np.concatenate((np.reshape(subj1_X[:,11], (subj1_X.shape[0],1)), 
# np.reshape(subj1_X[:,1], (subj1_X.shape[0],1)),
# np.reshape(subj1_X[:,6], (subj1_X.shape[0],1)),
# np.reshape(subj1_X[:,8], (subj1_X.shape[0],1)),
# np.reshape(subj1_X[:,10], (subj1_X.shape[0],1)),
# np.reshape(subj1_X[:,5], (subj1_X.shape[0],1)),
# np.reshape(subj1_X[:,9], (subj1_X.shape[0],1)),
# np.reshape(subj1_X[:,20], (subj1_X.shape[0],1))), axis=1)
# subj1Y_test = subj1_Y
########################################################################################

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb, val_list = NGBRegressor(n_estimators=500).fit(X_train, Y_train, X_test, Y_test)
import pickle
from pathlib import Path

file_path = Path.home()/'Documents'/'_UM'/'23Fall'/'EECS553'/'mlboost2023'/'ngbtest_4.p'

with file_path.open("wb") as f:
    pickle.dump(ngb, f)
