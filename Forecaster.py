from Models import LassoForecaster, ArimaForecaster, GBMForecaster
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Data paths
DATA_PATH_SP = "data/SP500_aug_15to20.csv"
DATA_PATH_HSI = "data/HSI_aug_15to20.csv"

""" Model parameters"""
PARAMS = {
    'START_DATE': "2015-08-01"

}

raw_data = pd.read_csv(DATA_PATH_HSI)
# Start simulation
forecaster = GBMForecaster(DATA_PATH_HSI)

forecaster.fit(raw_data.loc[:int(0.7 * raw_data.shape[0])])
mse = forecaster.validate(raw_data.loc[int(0.7 * raw_data.shape[0]):])
# forecaster.results.plot_predict(50, 2000)
# plt.show()
# ## GBM ##
# conf_int = forecaster.confidence_interval()
# print(conf_int[1][365], conf_int[0][365])
