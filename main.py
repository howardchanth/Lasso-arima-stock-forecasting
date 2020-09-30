from tester import RunsTester, StationarityTester
from models import LassoForecaster, ArimaForecaster, GBMForecaster
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Data paths
DATA_PATH_SP = "data/SP500_aug_15to20.csv"
DATA_PATH_HSI = "data/HSI_aug_15to20.csv"

""" Model parameters"""
PARAMS = {
    # TODO: Exclude weekends in the prediction plot
    'START_DATE': "2015-08-17",
    "END_DATE": "2020-07-20",
    "SERIES_NAME": "HSI",
    "SCENE_SIZE": 1000,
    "ALPHA": 0.05,
    "PLOT_CI": True,
    "d": 0,
    # TODO: Optimize lags by cross validation
    "LAG": 3
}
# --------------------------------------------------
#                    Load Data
# --------------------------------------------------
# Read data from path
if PARAMS['SERIES_NAME'] == "HSI":
    raw_data = pd.read_csv(DATA_PATH_HSI)
elif PARAMS['SERIES_NAME'] == "SP500":
    raw_data = pd.read_csv(DATA_PATH_SP)
else:
    ValueError("Invalid data name. Unable to load series.")

# Incorporate starting and ending date
start = raw_data['Date'][raw_data['Date'] == PARAMS['START_DATE']].index[0]
end = raw_data['Date'][raw_data['Date'] == PARAMS['END_DATE']].index[0]
# raw_data = raw_data.set_index('Date')
raw_data = raw_data[start:end]

series = raw_data['Adj Close']
# --------------------------------------------------
#                     Testing
# --------------------------------------------------
# Perform Runs Test on the randomness of the data
runs_tester = RunsTester(series)
runs_tester.test(PARAMS['ALPHA'])

# Perform augmented Dicky-Fuller test
stationarity_tester = StationarityTester(series, PARAMS)
adf = stationarity_tester.test()
# Use Dicky-Fuller test to obtain d
d = stationarity_tester.get_d()
# Update d to parameters
PARAMS['d'] = d

# --------------------------------------------------
#                    Simulation
# --------------------------------------------------

# Initialize forecaster
forecaster = GBMForecaster(PARAMS)

# Fitting the model
train_test_sep = int(0.7 * raw_data.shape[0])
forecaster.fit(raw_data.loc[:train_test_sep])

# Model summary
forecaster.summary()

# Model validation
mse = forecaster.validate(raw_data.loc[train_test_sep:])

# Plot the forecasting results
forecaster.plot(365, plot_ci=PARAMS['PLOT_CI'], alpha=PARAMS['ALPHA'])

# ## GBM ##
# print(conf_int[1][365], conf_int[0][365])
