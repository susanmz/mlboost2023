from ngboost import NGBRegressor

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from ngboost.scores import LogScore
from matplotlib import pyplot as plt
from early_stopping import earlyNGBRegressor
from plot_loss import plotLoss

## Boston
title = "Boston"
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # data
Y = raw_df.values[1::2, 2]  # target
iters = 1000

## Concrete
# title = "Concrete"
# df = pd.read_excel('.//data//concrete.xls')
# df.info()
# X = df.drop(['csMPa'], axis=1)
# Y = df['csMPa']
# X = X.to_numpy()
# Y = Y.to_numpy()
# iters = 1000

## Energy - Paper looks to have used Heating Load only
# title = "Energy"
# df = pd.read_excel('.//data//energy.xlsx')
# df.info()
# X = df.drop(['Y1', 'Y2'], axis=1)
# Y = df['Y1']  # Heating Load - Y1 (NLL is ~0.55, like paper); Cooling Load - Y2
# X = X.to_numpy()
# Y = Y.to_numpy()
# iters = 1000

## Yacht
# title = "Yacht"
# df = pd.read_csv('data//yacht.csv')
# df.info()
# X = df.drop(['Rr'], axis=1)
# Y = df['Rr']
# X = X.to_numpy()
# Y = Y.to_numpy()
# iters = 1000


## Power
# title = "Power"
# df = pd.read_excel('data//power.xlsx')
# df.info()
# X = df.drop(['PE'], axis=1)
# Y = df['PE']
# X = X.to_numpy()
# Y = Y.to_numpy()
# iters = 2000

## Wine
# title = "Wine"
# wine = fetch_ucirepo(id=186)
# X = wine.data.features
# Y = wine.data.targets
# X = X.to_numpy()
# Y = Y.to_numpy().ravel()
# iters = 1000

mse_arr = []
nll_arr = []
mse_best_itrs = []
nll_best_itrs = []

rounds = 5
earlyStopping = True
earlyStoppingType = "NLL"
plotLast = False

for i in range(rounds):
    # Split data
    X_nottest, X_test, y_nottest, y_test = train_test_split(X, Y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_nottest, y_nottest, test_size=0.2)

    ngb = earlyNGBRegressor(n_estimators=iters, learning_rate=0.01, Score=LogScore)

    if not earlyStopping:
        ngb.fit(X_nottest, y_nottest)

    if earlyStopping:
        ngb.set_early_stopping(type = earlyStoppingType, early_stopping_rounds = 200)
        ngb.fit(X_train, y_train, X_val, y_val)
        
        #pick the best iteration on the validation set
        y_preds = ngb.staged_predict(X_val)
        y_dists = ngb.staged_pred_dist(X_val)
        val_rmse = [mean_squared_error(y_pred, y_val, squared=False) for y_pred in y_preds]
        val_nll = [-y_dist.logpdf(y_val.flatten()).mean() for y_dist in y_dists]
        mse_best_itr = np.argmin(val_rmse)
        nll_best_itr = np.argmin(val_nll)

        # Extra info
        #print(mse_best_itr)
        #print(nll_best_itr)
        #print(ngb.best_val_loss_itr)

    if earlyStopping:
        ngb = NGBRegressor(n_estimators=ngb.best_val_loss_itr).fit(X_nottest, y_nottest)

    Y_preds = ngb.predict(X_test)
    Y_dists = ngb.pred_dist(X_test)    


    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, y_test, squared=False)
    print('Test MSE', test_MSE)

    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(y_test).mean()
    print('Test NLL', test_NLL)

    mse_arr.append(test_MSE)
    nll_arr.append(test_NLL)

    if earlyStopping:
        mse_best_itrs.append(mse_best_itr)
        nll_best_itrs.append(nll_best_itr)

print(f"RMSE of {rounds} rounds: ", np.around(np.mean(mse_arr),2), "±", np.around(np.std(mse_arr),2))
# print(mse_arr)
print(f"NLL of {rounds} rounds: ", np.around(np.mean(nll_arr),2), "±", np.around(np.std(nll_arr),2))
# print(nll_arr)

if plotLast:
    plotLoss(val_nll, val_rmse, nll_best_itr, mse_best_itr, save=True, title=title)