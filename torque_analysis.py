import scipy
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error

# Butterworth filter for torque trajectory smoothing
# 3rd order with cutoff frequency of 100Hz
def low_pass(data, cutoff, fs, order=3):
    nyq = 0.5 * 1000
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

stride_filepaths = [".//data//stroke_input_ml_s290.mat", 
".//data//stroke_input_ml_s163.mat",
".//data//stroke_input_ml_s161.mat",
".//data//stroke_input_ml_s122.mat",
".//data//stroke_input_ml_s67.mat"]
torque_filepaths = [".//data//stroke_output_ml_s290.mat",
".//data//stroke_output_ml_s163.mat",
".//data//stroke_output_ml_s161.mat",
".//data//stroke_output_ml_s122.mat",
".//data//stroke_output_ml_s67.mat"]
colors = ['b', 'g', 'r', 'c', 'm']

# file_path = Path.home()/'Documents'/'_UM'/'23Fall'/'EECS553'/'mlboost2023'/'ngbtest.p'
filepaths = ["C:\\Users\\susan\\Documents\\_UM\\23Fall\\EECS553\\mlboost2023\\ngbtest.p",
"C:\\Users\\susan\\Documents\\_UM\\23Fall\\EECS553\\mlboost2023\\ngbtest_1.p",
"C:\\Users\\susan\\Documents\\_UM\\23Fall\\EECS553\\mlboost2023\\ngbtest_2.p",
"C:\\Users\\susan\\Documents\\_UM\\23Fall\\EECS553\\mlboost2023\\ngbtest_3.p",
"C:\\Users\\susan\\Documents\\_UM\\23Fall\\EECS553\\mlboost2023\\ngbtest_4.p"
]


for i in range(len(stride_filepaths)):
    print(f"@@@@@@@@@@@@@@@@@@@@@THIS IS FOR SUBJECT {i}")
    test_rmse_all = []
    test_nll_all = []
    for file_path in filepaths:
        with open(file_path, "rb") as f:
            ngb_unpickled = pickle.load(f)
        stroke_data_all = scipy.io.loadmat(stride_filepaths[i])
        stroke_torque_all = scipy.io.loadmat(torque_filepaths[i])
        stroke_X_test = stroke_data_all['result']
        stroke_Y_test = stroke_torque_all['yout'][:,0]

        Y_preds = ngb_unpickled.predict(stroke_X_test, max_iter=200)
        Y_dists = ngb_unpickled.pred_dist(stroke_X_test, max_iter=200)

        # test Mean Squared Error
        test_MSE = mean_squared_error(Y_preds, stroke_Y_test, squared=False)
        # print('Test MSE', test_MSE)
        test_rmse_all.append(test_MSE)
        # test Negative Log Likelihood
        test_NLL = -Y_dists.logpdf(stroke_Y_test).mean()
        test_nll_all.append(test_NLL)
        # print('Test NLL', test_NLL)

    print(f'MEAN RMSE: {np.mean(test_rmse_all)}')
    print(f'STD RMSE: {np.std(test_rmse_all)}')
    print(f'MEAN NLL: {np.mean(test_nll_all)}')
    print(f'STD NLL: {np.std(test_nll_all)}')

    # Npoints = 101
    # norm_percent  = np.linspace(0,100,Npoints)
    # std = Y_dists.params['scale']
    # upper = Y_preds + (std * 1.96)
    # lower = Y_preds - (std * 1.96)

    # smooth_upper = low_pass(upper, cutoff=100, fs = 10000, order=3)
    # smooth_lower = low_pass(lower, cutoff=100, fs = 10000, order=3)
    # smooth_ypreds = low_pass(Y_preds, cutoff=100, fs = 10000, order=3)
    # plt.figure()
    # plt.fill_between(norm_percent, smooth_upper, smooth_lower, alpha=0.2, color=colors[i])
    # plt.plot(norm_percent, smooth_ypreds, color=colors[i])

    # plt.plot(norm_percent, stroke_Y_test, 'k')
    # plt.xlim([0, 100])
    # plt.xlabel('% Gait Cycle')
    # plt.ylabel('Torque (Nm/kg)')
    # plt.legend(['95% Confidence Interval', 'Predicted', 'Actual'], loc='lower right')
    # plt.title(f"Subject {i+1}")
    # plt.savefig(f".\\images\\Subject_{i+1}")
    # plt.show()

# plt.plot(norm_percent, Y_preds)
# plt.plot(norm_percent, stroke_Y_test)
# plt.xlabel('% Gait Cycle')
# plt.ylabel('Torque (Nm/kg)')
# plt.show()