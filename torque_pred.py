from ngboost import NGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

data_all_filepath = ".//data//data_all_K.mat"
torque_all_filepath = ".//data//torq_all_K.mat"

input_filepath = ".//data//input_ml.mat"
output_filepath = ".//data//output_ml.mat"

# Load the MATLAB file
data_all = scipy.io.loadmat(data_all_filepath)
torque_all = scipy.io.loadmat(torque_all_filepath)

input_strides = scipy.io.loadmat(input_filepath)
output_torque = scipy.io.loadmat(output_filepath)

# Access the variables stored in the MATLAB file
# Just level walking
# input_data = data_all['q_all'][:,:,:,0]  
# torque_data = torque_all['torq_all'][:,:,0] 
input_data = input_strides['result']
torque_data = output_torque['yout']

stride_test = input_data[:101,:]
torque_test = torque_data[:101,:]
X = input_data[101:,:]
Y = torque_data[101:,:]
################# Subject 1 ############################################
# subj1_X = input_data[:,:,0]  
# subj1_Y = torque_data[:,0]

# subj1X_test = np.concatenate((np.reshape(subj1_X[:,1], (subj1_X.shape[0],1)), subj1_X[:,5:7], subj1_X[:,8:12], np.reshape(subj1_X[:,20], (subj1_X.shape[0],1))), axis=1)
# subj1Y_test = subj1_Y

# train_input_data = input_data[:,:,1]
# ytrain_input_data = torque_data[:,1:]
# for i in range(2, len(input_data[0,0])):
#     # Vertically stack the arrays
#     train_input_data = np.concatenate((train_input_data, input_data[:,:,i]), axis=0)

# X = np.concatenate((np.reshape(train_input_data[:,1], (train_input_data.shape[0],1)), train_input_data[:,5:7], train_input_data[:,8:12], np.reshape(train_input_data[:,20], (train_input_data.shape[0],1))), axis=1)
# Y = np.reshape(ytrain_input_data, (ytrain_input_data.shape[0] * ytrain_input_data.shape[1],1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb, _ = NGBRegressor(n_estimators=1000).fit(X_train, Y_train, X_test, Y_test)
# ngb, _ = NGBRegressor(n_estimators=1000).fit(X, Y)
# Y_preds = ngb.predict(X_test)
# Y_dists = ngb.pred_dist(X_test)

# # test Mean Squared Error
# test_MSE = mean_squared_error(Y_preds, Y_test)
# print('Test MSE', test_MSE)

# # test Negative Log Likelihood
# test_NLL = -Y_dists.logpdf(Y_test).mean()
# print('Test NLL', test_NLL)

# plt.plot(Y_preds)
# plt.show()

# subj1_Y_preds = ngb.predict(subj1X_test, max_iter=ngb.best_val_loss_itr)
subj1_Y_preds = ngb.predict(stride_test)
plt.plot(subj1_Y_preds)
plt.plot(torque_test)
plt.show()