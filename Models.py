import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso

from statsmodels.graphics.tsaplots import plot_pacf
import pmdarima as pm


class Forecaster(ABC):
    """Abstract class for financial forecaster"""

    def __init__(self, data_path):
        """Initialize forecaster"""

        # Reading the stock price data
        self.data_path = data_path
        self.raw_data = pd.read_csv(data_path)
        self.series = self.raw_data['Adj Close']

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def conf_int(self, n_period):
        pass

    def validate(self, test):
        pass

    def plot(self):
        pass


class GBMForecaster(Forecaster):
    """
        Parameter Definitions
        Ref:
        https://towardsdatascience.com/simulating-stock-prices-in-python-using-geometric-brownian-motion-8dfd6e8c6b18

        S_0   :   initial stock price
        dt    :   time increment -> a day in our case
        T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
        N     :   number of time points in the prediction time horizon -> T/dt
        t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
        mu    :   mean of historical daily returns
        sigma :   standard deviation of historical daily returns
        b     :   array for brownian increments
        W     :   array for brownian path
    """

    def __init__(self, data_path):
        super().__init__(data_path)

        # The stock price series
        self.S = self.raw_data['Adj Close']
        self.S_0 = self.S.iloc[-1]

        # The series of return of stocks
        # Return = (S_t+1 - S_t) / S_t
        self.returns = (self.S.loc[1:] - self.S.shift(1).loc[1:]) / self.S.shift(1).loc[1:]

        self.mu = np.mean(self.returns)
        self.sigma = np.std(self.returns)

        # The array of simulation results
        self.S_pred = None

    def fit(self, train):
        """Fit model using the training data"""

        self.S = train['Adj Close']
        self.S_0 = self.S.iloc[-1]

        # The series of return of stocks
        # Return = (S_t+1 - S_t) / S_t
        self.returns = (self.S.loc[1:] - self.S.shift(1).loc[1:]) / self.S.shift(1).loc[1:]

        self.mu = np.mean(self.returns)
        self.sigma = np.std(self.returns)

    def predict(self, scene_size, dur, dt=1):
        """
            Simulate over scenarios and create an array of predicted stock prices
            :param scene_size: number of scenarios
            :param dt: time increment -> a day in our case
            :param dur: length of the prediction time horizon(how many time points to predict, same unit with dt(days))
        """
        # N     :   number of time points in the prediction time horizon -> dur/dt
        # t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
        N = dur / dt
        t = np.arange(1, int(N) + 1)

        # b     :   array for brownian increments
        # W     :   array for brownian path

        b = {str(scene): np.random.normal(0, 1, int(N)) for scene in range(1, scene_size + 1)}
        W = {str(scene): b[str(scene)].cumsum() for scene in range(1, scene_size + 1)}

        # Compute drift and diffusion parameter
        drift = (self.mu - 0.5 * self.sigma ** 2) * t
        diffusion = {str(scene): self.sigma * W[str(scene)] for scene in range(1, scene_size + 1)}

        S_pred = np.array([self.S_0 * np.exp(drift + diffusion[str(scene)])
                           for scene in range(1, scene_size + 1)])
        # S_pred = np.insert(S_pred, 0, self.S_0, axis=1)

        return S_pred

    def validate(self, test):
        dur = test.shape[0]
        pred = self.predict(1000, dur)

        mse = np.average((self.det_proj(pred) - test['Adj Close']) ** 2)

        return mse

    def det_proj(self, pred):
        """Deterministic projection"""
        return np.mean(pred, axis=0)

    def conf_int(self, n_period):
        # TODO: Need to double confirm whether naively sorting the array is appropriate
        sorted_price = np.sort(self.S_pred, axis=0)

        # Upper and lower limits
        upper = np.quantile(sorted_price, 0.05, axis=0)
        lower = np.quantile(sorted_price, 0.95, axis=0)

        return lower, upper

    def plot(self):
        series_pred = self.det_proj()
        days = np.arange(len(series_pred))
        lower, upper = self.conf_int()

        fig, ax = plt.subplots()
        ax.plot(days, series_pred)
        ax.fill_between(days, lower, upper, color='b', alpha=.1)
        plt.ylabel('Stock Prices')
        plt.xlabel('Prediction Days')
        plt.show()


class LassoForecaster(Forecaster):

    def __init__(self, data_path):
        super().__init__(data_path)

        # Grid Search parameters
        self.parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
        self.predictors = self.raw_data.drop(['Adj Close', 'Date'], axis=1)
        self.responses = self.raw_data['Adj Close'].values.reshape(-1, 1)
        self.model = Lasso()
        self.regressor = GridSearchCV(self.model, self.parameters, scoring='neg_mean_squared_error', cv=5)

    def fit(self, *args, **kwargs):
        self.regressor.fit(self.predictors, self.responses)

    def conf_int(self, n_period):
        pass


class ArimaForecaster(Forecaster):

    def __init__(self, data_path):
        super().__init__(data_path)
        self.model = pm.auto_arima(self.series, start_p=1, start_q=1,
                                   max_p=3, max_q=3, m=12,
                                   start_P=0, seasonal=False,
                                   d=1, D=1, trace=True,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)
        self.results = None

    def plot_acf(self):
        """Show the plot of ACF"""
        pd.plotting.autocorrelation_plot(self.series)
        plt.show()

    def plot_pacf(self):
        """Show the plot of PACF"""
        plot_pacf(self.series['Adj Close'])
        plt.show()

    def fit(self):
        self.results = self.model.fit(disp=0)

    def conf_int(self, n_period):
        _, conf_int = self.model.predict(n_period, return_conf_int=True, alpha=0.05)
        return conf_int

    def plot(self):
        pass
