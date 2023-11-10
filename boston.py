from ngboost import NGBRegressor

from sklearn.datasets import load_boston
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd

df = pd.read_excel('edited_Concrete_data.xls')

df.info()
X = df.drop(['csMPa'], axis=1)
Y = df['csMPa']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# boston = load_boston()
# wine = fetch_ucirepo(id=186)
# X = wine.data.features
# Y = wine.data.targets


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test).mean()
print('Test NLL', test_NLL)