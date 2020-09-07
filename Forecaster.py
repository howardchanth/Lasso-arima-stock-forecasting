from Models import LassoForecaster, ArimaForecaster, GBMForecaster

# Data paths
DATA_PATH_SP = "data/SP500_aug_15to20.csv"
DATA_PATH_HSI = "data/HSI_aug_15to20.csv"

""" Model parameters"""
PARAMS = {

}

# Start simulation
forecaster = GBMForecaster(DATA_PATH_SP)
forecaster.fit(1000)
conf_int = forecaster.confidence_interval()
forecaster.plot()

print(conf_int[1][365], conf_int[0][365])
