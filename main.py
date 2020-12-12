from tester import RunsTester, StationarityTester
from models import LassoForecaster, ArimaForecaster, GBMForecaster
import pandas as pd
import numpy as np

import datetime

import yfinance as yf

import warnings

warnings.filterwarnings("ignore")

# Data paths
DATA_PATHS = {
    "^GSPC": "data/HSI_aug_15to20.csv",
    "^HSI": "data/HSI_aug_15to20.csv"
}

""" Model parameters"""
PARAMS = {
    'START_DATE': "2015-08-17",
    "END_DATE": "2020-07-17",
    'PRED_END_DATE': "2021-07-21",
    "PRED_DUR": 365,
    "SERIES_NAME": "^HSI",  # ^GSPC: S&P500 Index; ^HSI: HSI Index
    "IS_REALTIME": False,
    "MODEL_NAME": "GBM",  # Lasso; Ridge; GBM; ARIMA
    "SCENE_SIZE": 1000,
    "ALPHA": 0.05,  # Level of significance
    "PLOT_CI": True,
    # Lasso-based ARIMA specific settings
    # Orders from traditional ARIMA model
    "d": 0,  # Initial rate of differencing parameter
    "LAG": 1,  # Lag of the ARIMA model (p)
    "MA_ORDER": 0  # Order of the MA series (q)
}
# --------------------------------------------------
#                    Load Data
# --------------------------------------------------

# Read data from path
if PARAMS['IS_REALTIME']:
    raw_data = yf.Ticker(PARAMS["SERIES_NAME"])
    raw_data = raw_data.history(period="1d",
                                interval="1d",
                                start=PARAMS['START_DATE'],
                                end=PARAMS['END_DATE']
                                ).reset_index()
else:
    try:
        raw_data = pd.read_csv(DATA_PATHS[PARAMS['SERIES_NAME']])
        # Incorporate starting and ending date
        start = raw_data['Date'][raw_data['Date'] == PARAMS['START_DATE']].index[0]
        end = raw_data['Date'][raw_data['Date'] == PARAMS['END_DATE']].index[0]

        raw_data = raw_data[start:end]
        raw_data = raw_data.reset_index().drop(['index'], axis=1)

    except ValueError:
        ValueError("Invalid data name. Unable to load series.")


raw_data = raw_data.reset_index().drop(['Date'], axis=1)
series = raw_data['Close']

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
# TODO: Choose the optimal part from validation
# Initialize forecaster
if PARAMS['MODEL_NAME'] == "Lasso" or PARAMS['MODEL_NAME'] == "Ridge":
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
duration = (datetime.datetime.strptime(PARAMS['PRED_END_DATE'], "%Y-%m-%d") -
            datetime.datetime.strptime(PARAMS['END_DATE'], "%Y-%m-%d")).days
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
