import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lasso

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso

DATAPATH_SP = "data/SP500_aug_15to20.csv"
DATAPATH_HSI = "data/HSI_aug_15to20.csv"
# Read raw data of S&P 500 and HSI
raw_data_sp = pd.read_csv(DATAPATH_SP)
raw_data_hsi = pd.read_csv(DATAPATH_HSI)

# Separate raw data into predictors and response (dropping the dates
predictors_sp = raw_data_sp.drop(['Close', 'Date'], axis=1)
close_sp = raw_data_sp['Close'].values.reshape(-1, 1)

predictors_hsi = raw_data_sp.drop(['Close', 'Date'], axis=1)
close_hsi = raw_data_sp['Close'].values.reshape(-1, 1)

# Grid search for fitting and tuning Lasso
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

lasso = Lasso()
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(predictors_hsi, close_hsi)

# Print best parameters
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
