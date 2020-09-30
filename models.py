import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_perocessing import DataProcessor
from abc import ABC, abstractmethod

# Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso

# ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
import pmdarima as pm


class Forecaster(ABC):
    """Abstract class for financial forecaster"""

    def __init__(self, params, name):
        """Initialize forecaster"""
        # Name of the forecaster
        self.name = name
        # Dictionary of Hyper-parameters
        self.params = params
        self.scene_size = params['SCENE_SIZE']

        # Historical stock price to be retrieved from the training data
        self.history = None

    @abstractmethod
    def fit(self, train):
        pass

    def predict(self, test):
        pass

    def conf_int(self, n_period, alpha):
        pass

    def validate(self, test):
        pass

    def plot(self, n_period, print_ci=False, alpha=0.05):
        pass

    def summary(self):
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

    def __init__(self, params, name='GBM'):
        super().__init__(params, name)

        # Model parameters
        self.mu = None
        self.sigma = None

        # Starting point of prediction
        self.s0 = 0

    def fit(self, train):
        """
        Fit the model and set up the initial value of prediction (s0)
        Set up the parameters mu and sigma
        :param train: A dataframe of training stock price and its predictors
        :return: None
        """

        # Fetch stock price series and most recent stock price
        s = train['Adj Close']
        self.s0 = s.iloc[-1]
        # Update stock price history
        self.history = s
        # The series of return of stocks
        # Return = (S_t+1 - S_t) / S_t
        returns = (s.loc[1:] - s.shift(1).loc[1:]) / s.shift(1).loc[1:]

        # Fit the model parameters
        self.mu = np.mean(returns)
        self.sigma = np.std(returns)

    def predict(self, dur, dt=1):
        # TODO: Redesign prediction strategy fetching the path having the least MSE
        """
            Simulate over scenarios and create an array of predicted stock prices
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
        diffusion = {str(scene): self.sigma * w[str(scene)] for scene in range(1, self.scene_size + 1)}

        s_pred = np.array([self.s0 * np.exp(drift + diffusion[str(scene)])
                           for scene in range(1, self.scene_size + 1)])
        return s_pred
        # S_pred = np.insert(S_pred, 0, self.S_0, axis=1)

    def validate(self, test):
        dur = test.shape[0]
        pred = self.predict(dur)

        mse = np.average((self.det_proj(pred) - test['Adj Close']) ** 2)

        return mse

    @staticmethod
    def det_proj(pred):
        """
        Deterministic projection, take mean of stock price over the scenarios
        """
        return np.mean(pred, axis=0)

    def conf_int(self, n_period, alpha):
        pred = self.predict(n_period)
        # Select the scenarios with (alpha)th quantile and (1-alpha)th quantile of stock price
        pred_mean = np.mean(pred, axis=0)
        pred_mean_sorted = np.sort(pred_mean)
        # Fetch the scenario number of (alpha)th quantile and (1-alpha)th quantile
        lower = pred_mean_sorted[round((len(pred_mean_sorted) - 1) * alpha)]
        upper = pred_mean_sorted[round((len(pred_mean_sorted) - 1) * (1 - alpha))]
        lower = np.where(pred_mean == lower)[0][0]
        upper = np.where(pred_mean == upper)[0][0]

        # Obtain the stock (alpha)th quantile and (1-alpha)th stock price series with the index obtained
        lower = pred[lower]
        upper = pred[upper]

        return lower, upper

    def plot(self, n_period, plot_ci=False, alpha=0.05):
        """
        Function to produce the confidence interval of future stock price
        :param n_period: Number of period of prediction
        :param plot_ci: Whether plot the confidence interval or not
        :param alpha: Significance level of confidence interval
        :return: None
        """
        # Randomly take a path generated
        pred = self.predict(n_period)[0]
        lower, upper = self.conf_int(n_period, alpha)

        # Combine history and prediction to the series to be plotted
        history = np.array(self.history)
        prices = np.append(history, pred)

        # History days from the history series
        n_hist_period = len(self.history)
        pred_days = np.arange(n_period)
        days = np.arange(- n_hist_period, n_period)

        fig, ax = plt.subplots()
        ax.plot(days, prices)
        if plot_ci:
            ax.fill_between(pred_days, lower, upper, color='b', alpha=.1)
        plt.ylabel('Stock Prices')
        plt.xlabel('Days')
        plt.savefig(f"./plots/{self.params['SERIES_NAME']}_{self.name}_Price.png")
        plt.show()

    def summary(self):
        print("-" * 45)
        print(" " * 10 + "Geometric Brownian Motion" + " " * 10)
        print("-" * 45)

        print(f"The drift parameter mu is: {self.mu}")
        print(f"The diffusion parameter sigma is: {self.sigma}")
        return


class LassoForecaster(Forecaster):

    def __init__(self, params, name='Lasso'):
        super().__init__(params, name)
        # Stock price at the beginning of prediction
        self.s0 = 0
        self.lag = params['LAG']

        # Initialize the Lasso model
        # Grid Search parameters
        self.parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
        self.predictors = None
        self.model = Lasso()
        self.regressor = GridSearchCV(self.model, self.parameters, scoring='neg_mean_squared_error', cv=10)

    def fit(self, train):
        # TODO: Incorporate refit in
        """
        Function fitting the model and set up the initial value of prediction
        Update historical stock price
        :param train: A dataframe of training stock price and its predictors
        :return: None
        """
        # Copy stock price info
        s = train['Adj Close']
        self.s0 = s.iloc[-1]

        # Convert predictors to time series predictors
        # If fitting first time
        if self.predictors is None:
            raw_predictors = train.drop(['Adj Close', 'Date'], axis=1)
            dp = DataProcessor(raw_predictors, self.params)
            predictors = dp.get_lasso_predictor()
            # Store time series predictors
            self.predictors = predictors
        else:
            predictors = self.predictors

        # Truncate the response data to length n - lag to incorporate the predictors
        responses = train['Adj Close'].values[self.lag:].reshape(-1, 1)

        # Number of predictors
        self.regressor.fit(predictors, responses)

    def validate(self, test):
        """
        :param test: A dataframe of testing stock price and its predictors
        :return: Testing MSE
        """
        dur = test.shape[0]
        responses = test['Adj Close'].values.reshape(-1, 1)
        pred = self.predict(dur)
        mse = np.average((pred - responses) ** 2)
        return mse

    def predict(self, dur):
        # TODO: Decide whether n_period or testing data
        # Create copy of regressor and predictors
        predictors = self.predictors
        regressor = self.regressor
        prediction = pd.Series([])

        for _ in range(dur):
            # Make prediction of (n-lag) vector
            # TODO: Fix the bug by completing the refit and predict function
            pred = regressor.predict(predictors)
            prediction.append(pd.Series(pred[-1]))
            predictors = np.column_stack([predictors, pred])
            # collect predicted stock price
            prediction = pred[-1]

        return prediction

    def conf_int(self, n_period, alpha):
        # TODO: Investigate how to represent the CI for Lasso. Bootstrapping?
        """
        Function computing the confidence interval of the Lasso regression
        :param n_period: Number of period of prediction
        :param alpha: Significance level of confidence interval
        :return: None
        """
        pass

    def plot(self, n_period, plot_ci=False, alpha=0.05):
        # TODO: Design plotting for Lasso
        pass

    def summary(self):
        print("-" * 50)
        print(" " * 19 + "Lasso Model" + " " * 19)
        print("-" * 50)

        print(f"The best parameter is: {self.regressor.best_params_}")
        print(f"The best estimators are: {self.regressor.best_estimator_}")


class ArimaForecaster(Forecaster):

    def __init__(self, params, name='ARIMA'):
        super().__init__(params, name)

        # Number of differencing needed
        self.d = params['d']

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
        # Record stock price series
        self.history = series
        self.model = pm.auto_arima(series, start_p=1, start_q=1,
                                   max_p=3, max_q=3, m=12,
                                   start_P=0, seasonal=False,
                                   d=self.d, D=1, trace=True,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)

        self.results = self.model.fit(series, disp=0)

    def predict(self, n_period):
        pred = self.model.predict(n_period)
        return pred

    def validate(self, test):
        n_period = test.shape()[0]
        pred = self.predict(n_period)
        responses = test['Adj Close'].values.reshape(-1, 1)
        mse = np.average((pred - responses) ** 2)
        return mse

    def conf_int(self, n_period, alpha=0.05):
        """
        Function producing the confidence interval of future stock price
        :param n_period: Number of period of prediction
        :param alpha: Significance level of confidence interval
        :return: None
        """
        _, conf_int = self.model.predict(n_period, return_conf_int=True, alpha=alpha)
        return conf_int

    def plot(self, n_period, plot_ci=False, alpha=0.05):
        # TODO: investigate why straight line
        """
        Function to produce the confidence interval of future stock price
        :param n_period: Number of period of prediction
        :param plot_ci: Whether plot the confidence interval or not
        :param alpha: Significance level of confidence interval
        :return: None
        """
        # Fetch prediction and confidence intervals
        pred, conf_int = self.model.predict(n_period, return_conf_int=True, alpha=alpha)
        history = np.array(self.history)
        prices = np.append(history, pred)

        # History days from the history series
        n_hist_period = len(self.history)
        pred_days = np.arange(n_period)
        days = np.arange(- n_hist_period, n_period)

        lower = conf_int[:, 0]
        upper = conf_int[:, 1]
        fig, ax = plt.subplots()
        ax.plot(days, prices)
        if plot_ci:
            ax.fill_between(pred_days, lower, upper, color='r', alpha=.1)
        plt.ylabel('Stock Prices')
        plt.xlabel('Prediction Days')
        plt.savefig(f"./plots/{self.params['SERIES_NAME']}_{self.name}_Price.png")
        plt.show()

    def summary(self):
        print(self.summary())
