# Ignore warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from tester import RunsTester, StationarityTester
from models import LassoForecaster, ArimaForecaster, GBMForecaster
import pandas as pd
import numpy as np

# TODO: Incorporate realtime prediction
import yfinance as yf
import matplotlib.pyplot as plt

# Data paths
DATA_PATH_SP = "data/SP500_aug_15to20.csv"
DATA_PATH_HSI = "data/HSI_aug_15to20.csv"

""" Model parameters"""
PARAMS = {
    # TODO: Exclude weekends in the prediction plot
    'START_DATE': "2015-08-17",
    "END_DATE": "2020-07-17",
    'PRED_END_DATE': "2021-07-21",
    "PRED_DUR": 365,
    "SERIES_NAME": "SP500",
    "MODEL_NAME": "Lasso",
    "SCENE_SIZE": 100,
    "ALPHA": 0.05,
    "PLOT_CI": True,
    "d": 0,  # Initial rate of differencing parameter
    "LAG": 10  # Manual tuning
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

raw_data = raw_data[start:end]
raw_data = raw_data.reset_index().drop(['index'], axis=1)

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
if PARAMS['MODEL_NAME'] == "Lasso":
    forecaster = LassoForecaster(PARAMS)
elif PARAMS['MODEL_NAME'] == "GBM":
    forecaster = GBMForecaster(PARAMS)
else:
    forecaster = ArimaForecaster(PARAMS)

# Fitting the model
train_test_sep = int(0.7 * raw_data.shape[0])
forecaster.fit(raw_data.loc[:train_test_sep])

# Model summary
forecaster.summary()

# Forecasting one year stock price
prediction = forecaster.predict(PARAMS['PRED_DUR'])

# Model validation
mape, mse = forecaster.validate(raw_data.loc[train_test_sep:])
print("-" * 20 + "MSEs" + "-" * 20)
print(f"The MSE is: {mse}")
print(f"The RMSE is: {np.sqrt(mse)}")
print(f"The MAPE is: {mape}")
# --------------------------------------------------
#                     Plots
# --------------------------------------------------
# Plot the forecasting results
forecaster.plot(365, plot_ci=PARAMS['PLOT_CI'], alpha=PARAMS['ALPHA'])

if PARAMS['MODEL_NAME'] == "ARIMA":
    # Obtain the differenced series
    series_diff = series.diff().shift(-1).iloc[:-1]

    # Plot ACF and PACF
    forecaster.plot_acf(series_diff)
    forecaster.plot_pacf(series_diff)
