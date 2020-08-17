import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lasso

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

DATAPATH_SP = "data/SP500_aug_15to20.csv"
DATAPATH_HSI = "data/HSI_aug_15to20.csv"
# Read raw data of S&P 500 and HSI
raw_data_sp = pd.read_csv(DATAPATH_SP)
raw_data_hsi = pd.read_csv(DATAPATH_HSI)

#