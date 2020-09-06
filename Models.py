import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Forecaster(ABC):
    """Abstract class for financial forecaster"""

    def __init__(self, data_path):
        """Initialize forecaster by reading the data series"""
        self.data_path = data_path
        self.raw_data = pd.read_csv(data_path)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def predict(self):
        pass

    def confidence_interval(self):
        pass

    def validate(self):
        pass

    def plot(self):
        pass


class GBM(Forecaster):
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

    def fit(self, scene_size, dt=1, dur=365):
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

        self.S_pred = np.array([self.S_0 * np.exp(drift + diffusion[str(scene)])
                                for scene in range(1, scene_size + 1)])
        self.S_pred = np.insert(self.S_pred, 0, self.S_0, axis=1)

    def predict(self):
        """Deterministic projection"""
        return np.mean(self.S_pred, axis=1)

    def confidence_interval(self):
        # TODO: Need to double confirm whether naively sorting the array is appropriate
        sorted_price = np.sort(self.S_pred, axis=0)

        # Upper and lower limits
        upper = np.quantile(sorted_price, 0.05, axis=0)
        lower = np.quantile(sorted_price, 0.95, axis=0)

        return lower, upper

    def plot(self):
        # TODO: Plot stock price projection
        pass
