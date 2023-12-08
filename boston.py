from ngboost import NGBRegressor

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Boston
# title = "Boston"
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # data
# Y = raw_df.values[1::2, 2]  # target

## Concrete
# title = "Concrete"
# df = pd.read_excel('.//data//concrete.xls')
# df.info()
# X = df.drop(['csMPa'], axis=1)
# Y = df['csMPa']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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

## Power
title = "Power"
df = pd.read_excel('data//power.xlsx')
df.info()
X = df.drop(['PE'], axis=1)
Y = df['PE']

## Wine
# title = "Wine"
# wine = fetch_ucirepo(id=186)
# X = wine.data.features
# Y = wine.data.targets

mse_arr = []
nll_arr = []
for i in range(10):
    X_nottest, X_test, Y_nottest, Y_test = train_test_split(X, Y, test_size=0.1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_nottest, Y_nottest, test_size=0.2)

    ngb,val_loss_list = NGBRegressor(n_estimators=50).fit(X_train, Y_train, X_val, Y_val, early_stopping_rounds=1)
    # ngb,val_loss_list = NGBRegressor(n_estimators=50).fit(X_train, Y_train, X_val, Y_val, early_stopping_rounds=50)
    # ngb,val_loss_list = NGBRegressor().fit(X_train, Y_train, X_val = X_test, Y_val= Y_test)
    print(ngb.best_val_loss_itr)

    Y_preds = ngb.predict(X_test)
    Y_dists = ngb.pred_dist(X_test)

    # test Mean Squared Error
    test_MSE = mean_squared_error(Y_preds, Y_test)
    print('Test MSE', test_MSE)

    # test Negative Log Likelihood
    test_NLL = -Y_dists.logpdf(Y_test).mean()
    print('Test NLL', test_NLL)

    mse_arr.append(test_MSE)
    nll_arr.append(test_NLL)

print("MSE average of 10 rounds:", np.mean(mse_arr))
print("MSE std of 10 rounds:", np.std(mse_arr))
print(mse_arr)
print("NLL average of 10 rounds:", np.mean(nll_arr))
print("NLL std of 10 rounds:", np.std(nll_arr))
print(nll_arr)

# plt.plot(range(1, len(val_loss_list)+1), val_loss_list)
# plt.plot(ngb.best_val_loss_itr, val_loss_list[ngb.best_val_loss_itr - 1], 'ro')
# plt.xlabel("Iterations")
# plt.ylabel("Validation Loss")
# plt.title(title)
# fig_filename = title+'_3.png'
# plt.savefig(f'.//images//{fig_filename}')
# plt.show()