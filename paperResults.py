from ngboost import NGBRegressor

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from ngboost.scores import LogScore
from plot_loss import plotLoss

## Boston
title = "Boston"
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # data
Y = raw_df.values[1::2, 2]  # target

## Concrete
# title = "Concrete"
# df = pd.read_excel('.//data//concrete.xls')
# df.info()
# X = df.drop(['csMPa'], axis=1)
# Y = df['csMPa']


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
mse_best_itrs = []
nll_best_itrs = []

splits = 1
n = X.shape[0]
folds = []

for i in range(splits):
    permutation = np.random.choice(range(n), n, replace = False)
    end_train = round(n * 9.0 / 10)
    end_test = n

    train_index = permutation[0:end_train]
    test_index = permutation[end_train:n]
    folds.append((train_index, test_index))

for itr, (train_index, test_index) in enumerate(folds):

    X_trainall, X_test = X[train_index], X[test_index]
    y_trainall, y_test = Y[train_index], Y[test_index]

    #X_trainall, X_test = X.iloc[train_index].values, X.iloc[test_index].values
    #y_trainall, y_test = Y.iloc[train_index].values, Y.iloc[test_index].values

    X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)

    ngb = NGBRegressor(n_estimators=500, learning_rate=0.01, Score=LogScore).fit(X_train, y_train, X_val , y_val)

    # pick the best iteration on the validation set
    y_preds = ngb.staged_predict(X_val)
    y_dists = ngb.staged_pred_dist(X_val)

    val_rmse = [mean_squared_error(y_pred, y_val) for y_pred in y_preds]
    val_nll = [-y_dist.logpdf(y_val.flatten()).mean() for y_dist in y_dists]
    mse_best_itr = np.argmin(val_rmse)
    nll_best_itr = np.argmin(val_nll)

    print(nll_best_itr)
    print(ngb.best_val_loss_itr)
    
    # re-train using all the data after tuning number of iterations
    ngb = NGBRegressor(n_estimators=200, learning_rate=0.01, Score=LogScore).fit(X_trainall, y_trainall)

    # the final prediction for this fold
    Y_dist_MSE = ngb.pred_dist(X_test, max_iter=mse_best_itr)
    Y_dist_NLL = ngb.pred_dist(X_test, max_iter=nll_best_itr)

    test_MSE= np.sqrt(mean_squared_error(Y_dist_MSE.mean(), y_test))
    test_NLL= -Y_dist_NLL.logpdf(y_test.flatten()).mean()

    mse_arr.append(test_MSE)
    nll_arr.append(test_NLL)
    mse_best_itrs.append(mse_best_itr)
    nll_best_itrs.append(nll_best_itr)


print("MSE average of 20 rounds:", np.mean(mse_arr))
print("MSE std of 20 rounds:", np.std(mse_arr))
print("Min MSE:", np.min(mse_arr))
print("NLL average of 20 rounds:", np.mean(nll_arr))
print("NLL std of 20 rounds:", np.std(nll_arr))
print("Min NLL:", np.min(nll_arr))

print("MSE best iterations:", mse_best_itrs)
print("NLL best iterations:", nll_best_itrs)
plotLoss(val_nll, val_rmse, nll_best_itr, mse_best_itr, save=True, title="Concrete")
