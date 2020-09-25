import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso

# ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
import pmdarima as pm


class Forecaster(ABC):
    """Abstract class for financial forecaster"""

    def __init__(self, params):
        """Initialize forecaster"""
        # Dictionary of Hyper-parameters
        self.params = params
        self.scene_size = params['SCENE_SIZE']

    @abstractmethod
    def fit(self, train):
        pass

    def predict(self, test):
        pass

    def conf_int(self, n_period, alpha):
        pass

    def validate(self, test):
        pass

    def plot(self, n_period, alpha):
        pass


class GBMForecaster(Forecaster):
    """
        Parameter Definitions
        Ref:
        https://towardsdatascience.com/simulating-stock-prices-in-python-using-geometric-brownian-motion-8dfd6e8c6b18

        s0   :   initial stock price
        dt    :   time increment -> a day in our case
        T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
        N     :   number of time points in the prediction time horizon -> T/dt
        t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
        mu    :   mean of historical daily returns
        sigma :   standard deviation of historical daily returns
        b     :   array for brownian increments
        Ww    :   array for brownian path
    """

    def __init__(self, params):
        super().__init__(params)

        # Model parameters
        self.mu = None
        self.sigma = None

        # Starting point of prediction
        self.s0 = 0

    def fit(self, train):
        """Fit model using the training data"""

        # Fetch stock price series and most recent stock price
        s = train['Adj Close']
        self.s0 = s.iloc[-1]
        # The series of return of stocks
        # Return = (S_t+1 - S_t) / S_t
        returns = (s.loc[1:] - s.shift(1).loc[1:]) / s.shift(1).loc[1:]

        # Fit the model parameters
        self.mu = np.mean(returns)
        self.sigma = np.std(returns)

    def predict(self, dur, dt=1):
        # TODO: Put scene_size to global
        """
            Simulate over scenarios and create an array of predicted stock prices
            :param scene_size: number of scenarios
            :param dt: time increment -> a day in our case
            :param dur: length of the prediction time horizon(how many time points to predict, same unit with dt(days))
        """
        # n_period     :   number of time points in the prediction time horizon -> dur/dt
        # t            :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
        n_period = dur / dt
        t = np.arange(1, int(n_period) + 1)

        # b     :   array for brownian increments
        # W     :   array for brownian path

        b = {str(scene): np.random.normal(0, 1, int(n_period)) for scene in range(1, self.scene_size + 1)}
        w = {str(scene): b[str(scene)].cumsum() for scene in range(1, self.scene_size + 1)}

        # Compute drift and diffusion parameter
        drift = (self.mu - 0.5 * self.sigma ** 2) * t
        diffusion = {str(scene): self.sigma * w[str(scene)] for scene in range(1, scene_size + 1)}

        s_pred = np.array([self.s0 * np.exp(drift + diffusion[str(scene)])
                           for scene in range(1, self.scene_size + 1)])
        return s_pred
        # S_pred = np.insert(S_pred, 0, self.S_0, axis=1)

    def validate(self, test):
        dur = test.shape[0]
        pred = self.predict(1000, dur)

        mse = np.average((self.det_proj(pred) - test['Adj Close']) ** 2)

        return mse

    @staticmethod
    def det_proj(pred):
        """Deterministic projection"""
        return np.mean(pred, axis=0)

    def conf_int(self, n_period, alpha):
        # TODO: Redesign confidence interval
        pred = self.predict(1000, n_period)
        sorted_price = np.sort(pred, axis=0)

        # Upper and lower limits
        upper = np.quantile(sorted_price, alpha, axis=0)
        lower = np.quantile(sorted_price, 1 - alpha, axis=0)

        return lower, upper

    def plot(self, n_period, alpha):
        # TODO: Resolve discrepancies between n_period and that in predict
        pred = self.predict(1000, n_period)
        series_pred = self.det_proj(pred)
        days = np.arange(n_period)
        lower, upper = self.conf_int(n_period, alpha)

        fig, ax = plt.subplots()
        ax.plot(days, series_pred)
        ax.fill_between(days, lower, upper, color='b', alpha=.1)
        plt.ylabel('Stock Prices')
        plt.xlabel('Prediction Days')
        plt.show()


class LassoForecaster(Forecaster):

    def __init__(self, params):
        super().__init__(params)
        # Stock price at the beginning of prediction
        self.s0 = 0

        # Initialize the Lasso model
        # Grid Search parameters
        self.parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
        self.model = Lasso()
        self.regressor = GridSearchCV(self.model, self.parameters, scoring='neg_mean_squared_error', cv=5)

    def fit(self, train):
        """
        Function fitting the model and set up the initial value of prediction
        :param train: A dataframe of training stock price and its predictors
        :return: None
        """
        predictors = train.drop(['Adj Close', 'Date'], axis=1)
        responses = train['Adj Close'].values.reshape(-1, 1)
        s = train['Adj Close']
        self.s0 = s.iloc[-1]
        self.regressor.fit(predictors, responses)

    def validate(self, test):
        """
        :param test: A dataframe of testing stock price and its predictors
        :return: Testing MSE
        """
        predictors = test.drop(['Adj Close', 'Date'], axis=1)
        responses = test['Adj Close'].values.reshape(-1, 1)
        pred = self.regressor.predict(predictors)
        mse = np.average((pred - responses) ** 2)
        return mse

    def conf_int(self, n_period, alpha):
        pass


class ArimaForecaster(Forecaster):

    def __init__(self, params):
        super().__init__(params)

        self.model = None
        self.results = None

    @staticmethod
    def plot_acf(series):
        """
        Show the plot of ACF
        @param series: The stock price series
        """
        pd.plotting.autocorrelation_plot(series)
        plt.show()

    @staticmethod
    def plot_pacf(series):
        """
        Show the plot of PACF
        @param series: The stock price series
        """
        plot_pacf(series)
        plt.show()

    def fit(self, train):
        series = train['Adj Close']
        self.model = pm.auto_arima(series, start_p=1, start_q=1,
                                   max_p=3, max_q=3, m=12,
                                   start_P=0, seasonal=False,
                                   d=1, D=1, trace=True,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)

        self.results = self.model.fit(series, disp=0)

    def validate(self, test):
        n_period = test.shape[0]
        responses = test['Adj Close'].values.reshape(-1, 1)
        pred = self.model.predict(n_period)
        mse = np.average((pred - responses) ** 2)
        return mse

    def conf_int(self, n_period, alpha):
        """
        Fuction to produce the confidence interval of future stock price
        :param n_period: Number of period of prediction
        :param alpha: Significance level of confidence interval
        :return:
        """
        _, conf_int = self.model.predict(n_period, return_conf_int=True, alpha=alpha)
        return conf_int

    def plot(self, ):
        # TODO: Plot the predicted stock price series and the CIs
        pass
