import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Acknowledgement

# Raw data path
DATA_PATH_SP = "data/SP500_aug_15to20.csv"
DATA_PATH_HSI = "data/HSI_aug_15to20.csv"

# Read raw data of S&P 500 and HSI
raw_data_sp = pd.read_csv(DATA_PATH_SP)
raw_data_hsi = pd.read_csv(DATA_PATH_HSI)

# Preliminary Calculations

# The stock price series
S = raw_data_sp['Adj Close']

# The series of return of stocks
# Return = (S_t+1 - S_t) / S_t
returns = (S.loc[1:] - S.shift(1).loc[1:]) / S.shift(1).loc[1:]

# Parameter Definitions
# Ref: https://towardsdatascience.com/simulating-stock-prices-in-python-using-geometric-brownian-motion-8dfd6e8c6b18

# S_0   :   initial stock price
# dt    :   time increment -> a day in our case
# T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
# N     :   number of time points in the prediction time horizon -> T/dt
# t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
# mu    :   mean of historical daily returns
# sigma :   standard deviation of historical daily returns
# b     :   array for brownian increments
# W     :   array for brownian path

# Read the latest stock price as initial stock price
S_0 = raw_data_sp['Adj Close'].iloc[-1]
dt = 1  # 1 day
T = 365  # 1 year
N = T / dt
t = np.arange(1, int(N) + 1)
mu = np.mean(returns)
sigma = np.std(returns)

# simulate scenarios
scene_size = 3  # number of scenario
b = {str(scene): np.random.normal(0, 1, int(N)) for scene in range(1, scene_size + 1)}
W = {str(scene): b[str(scene)].cumsum() for scene in range(1, scene_size + 1)}

# Compute drift and diffusion parameter
drift = (mu - 0.5 * sigma ** 2) * t
diffusion = {str(scene): sigma * W[str(scene)] for scene in range(1, scene_size + 1)}

# Predict the future stock prices
S_pred = np.array([S_0 * np.exp(drift + diffusion[str(scene)]) for scene in range(1, scene_size + 1)])
S_pred = np.insert(S_pred, 0, S_0, axis=1)

# Calculate the mean of scenarios
S_pred_mean = np.mean(S_pred, axis=1)

# Compute confidence interva
#
# l
