from ngboost import NGBRegressor

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ngboost.scores import LogScore

## Boston
# title = "Boston"
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # data
# Y = raw_df.values[1::2, 2]  # target

## Concrete
title = "Concrete"
df = pd.read_excel('.//data//concrete.xls')
df.info()
X = df.drop(['csMPa'], axis=1)
Y = df['csMPa']


## Energy - Paper looks to have used Heating Load only
# title = "Energy"
# df = pd.read_excel('.//data//energy.xlsx')
# df.info()
# X = df.drop(['Y1', 'Y2'], axis=1)
# Y = df['Y1']  # Heating Load - Y1 (NLL is ~0.55, like paper); Cooling Load - Y2

## Yacht
# title = "Yacht"
# df = pd.read_csv('data//yacht.csv')
# df.info()
# X = df.drop(['Rr'], axis=1)
# Y = df['Rr']
# X_nottest, X_test, Y_nottest, Y_test = train_test_split(X, Y, test_size=0.1)
# X_train, X_val, Y_train, Y_val = train_test_split(X_nottest, Y_nottest, test_size=0.2)


## Power
# title = "Power"
# df = pd.read_excel('data//power.xlsx')
# df.info()
# X = df.drop(['PE'], axis=1)
# Y = df['PE']
# X_nottest, X_test, Y_nottest, Y_test = train_test_split(X, Y, test_size=0.1)
# X_train, X_val, Y_train, Y_val = train_test_split(X_nottest, Y_nottest, test_size=0.2)

## Wine
# title = "Wine"
# wine = fetch_ucirepo(id=186)
# X = wine.data.features
# Y = wine.data.targets

mse_arr = []
nll_arr = []

splits = 20
n = X.shape[0]
np.random.seed(1)
folds = []

for i in range(splits):
    permutation = np.random.choice(range(n), n, replace = False)
    end_train = round(n * 9.0 / 10)
    end_test = n

    train_index = permutation[0:end_train]
    test_index = permutation[end_train:n]
    folds.append((train_index, test_index))

for itr, (train_index, test_index) in enumerate(folds):

    #X_trainall, X_test = X[train_index], X[test_index]
    #y_trainall, y_test = Y[train_index], Y[test_index]

    X_trainall, X_test = X.iloc[train_index].values, X.iloc[test_index].values
    y_trainall, y_test = Y.iloc[train_index].values, Y.iloc[test_index].values


    X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)

    ngb = NGBRegressor(n_estimators=2000, learning_rate=0.01, Score=LogScore).fit(X_train, y_train, X_val , y_val)

    # pick the best iteration on the validation set
    y_preds = ngb.staged_predict(X_val)
    y_forecasts = ngb.staged_pred_dist(X_val)

    val_rmse = [mean_squared_error(y_pred, y_val) for y_pred in y_preds]
    val_nll = [-y_forecast.logpdf(y_val.flatten()).mean() for y_forecast in y_forecasts]
    best_itr = np.argmin(val_rmse) + 1

    # re-train using all the data after tuning number of iterations
    ngb = NGBRegressor(n_estimators=2000, learning_rate=0.01, Score=LogScore).fit(X_trainall, y_trainall)

    # the final prediction for this fold
    Y_dists = ngb.pred_dist(X_test, max_iter=best_itr)

    test_MSE= np.sqrt(mean_squared_error(Y_dists.mean(), y_test))
    test_NLL= -Y_dists.logpdf(y_test.flatten()).mean()

    mse_arr.append(test_MSE)
    nll_arr.append(test_NLL)


print("MSE average of 20 rounds:", np.mean(mse_arr))
print("MSE std of 20 rounds:", np.std(mse_arr))
print("Min MSE:", np.min(mse_arr))
print("NLL average of 20 rounds:", np.mean(nll_arr))
print("NLL std of 20 rounds:", np.std(nll_arr))
print("Min NLL:", np.min(nll_arr))



    


# for i in range(20):
#     X_nottest, X_test, Y_nottest, Y_test = train_test_split(X, Y, test_size=0.1)
#     X_train, X_val, Y_train, Y_val = train_test_split(X_nottest, Y_nottest, test_size=0.2)

#     # No early stopping
    
#     ngb,val_loss_list = NGBRegressor().fit(X_train, Y_train, X_val, Y_val)
#     Y_preds = ngb.predict(X_test)
#     Y_dists = ngb.pred_dist(X_test)

#     # Early stopping

#     #ngb,val_loss_list = NGBRegressor(n_estimators=500).fit(X_train, Y_train, X_val, Y_val, early_stopping_rounds=50)
#     #Y_preds = ngb.predict(X_test, max_iter=ngb.best_val_loss_itr)
#     #Y_dists = ngb.pred_dist(X_test, max_iter=ngb.best_val_loss_itr)
#     #print(ngb.best_val_loss_itr)    


#     # test Mean Squared Error
#     test_MSE = mean_squared_error(Y_preds, Y_test)
#     #print('Test MSE', test_MSE)

#     # test Negative Log Likelihood
#     test_NLL = -Y_dists.logpdf(Y_test).mean()
#     #print('Test NLL', test_NLL)

#     mse_arr.append(test_MSE)
#     nll_arr.append(test_NLL)

# print("MSE average of 20 rounds:", np.mean(mse_arr))
# print("MSE std of 20 rounds:", np.std(mse_arr))
# #print(mse_arr)
# print("NLL average of 20 rounds:", np.mean(nll_arr))
# print("NLL std of 20 rounds:", np.std(nll_arr))
# #print(nll_arr)

# plt.plot(range(1, len(val_loss_list)+1), val_loss_list)
# plt.plot(ngb.best_val_loss_itr, val_loss_list[ngb.best_val_loss_itr - 1], 'ro')
# plt.xlabel("Iterations")
# plt.ylabel("Validation Loss")
# plt.title(title)
# fig_filename = title+'_3.png'
# plt.savefig(f'.//images//{fig_filename}')
# plt.show()