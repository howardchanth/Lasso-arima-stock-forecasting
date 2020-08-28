import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

# Data paths
DATAPATH_SP = "data/SP500_aug_15to20.csv"
DATAPATH_HSI = "data/HSI_aug_15to20.csv"

# Read raw data of S&P 500 and HSI
raw_data_sp = pd.read_csv(DATAPATH_SP)
raw_data_hsi = pd.read_csv(DATAPATH_HSI)

# Date parser
def parser(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')

# Load series
series = pd.read_csv(DATAPATH_HSI, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# Plot ACF to find AR lag (lag = 0)
pd.plotting.autocorrelation_plot(series)
plt.show()

# Plot PACF to find MA lag (lag = 3)
plot_pacf(series['Adj Close'])
plt.show()

# Fit ARIMA model
model = ARIMA(series['Adj Close'], order=(0,2,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())