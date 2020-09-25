from tester import RunTester
from models import LassoForecaster, ArimaForecaster, GBMForecaster
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Data paths
DATA_PATH_SP = "data/SP500_aug_15to20.csv"
DATA_PATH_HSI = "data/HSI_aug_15to20.csv"

""" Model parameters"""
PARAMS = {
    'START_DATE': "2015-08-01",
    "END_DATE": "2021-08-01",
    "SCENE_SIZE": 1000
}

# Read data from path
raw_data = pd.read_csv(DATA_PATH_SP)
series = raw_data['Adj Close']

# Perform Runs Test on the randomness of the data
run_tester = RunTester(series)
run_tester.test(0.05)

# Start simulation
forecaster = LassoForecaster(PARAMS)

# Fitting the model
train_test_sep = int(0.7 * raw_data.shape[0])
forecaster.fit(raw_data.loc[:train_test_sep])

# Model validation
mse = forecaster.validate(raw_data.loc[train_test_sep:])

# forecaster.plot(365, 0.05)

# ## GBM ##
# print(conf_int[1][365], conf_int[0][365])
