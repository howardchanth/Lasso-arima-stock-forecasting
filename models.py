import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

# ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
        self.dur = params['PRED_DUR']  # Prediction duration

        # Historical stock price to be retrieved from the training data
        self.history = None

    @abstractmethod
    def fit(self, train):
        pass

    def predict(self, *args, **kwargs):
        """
        Dummy method
        :param test: Testing dataframe
        :return: A series/dataframe of predictions
        """
        return

    def conf_int(self, n_period, alpha):
        """
        Dummy method
        :param n_period: Length of prediction period
        :param alpha: Significance level of the test
        :return: lower, upper: The lower and upper quantiles
        """
        lower = None
        upper = None
        return lower, upper

    def validate(self, test):
        pass

    def plot(self, n_period, plot_ci=False, alpha=0.05):
        """
        Function to produce the confidence interval of future stock price
        :param n_period: Number of period of prediction
        :param plot_ci: Whether plot the confidence interval or not
        :param alpha: Significance level of confidence interval
        :return: None
        """
        # Fetch prediction results
        pred = self.predict(n_period)

        if plot_ci:
            lower, upper = self.conf_int(n_period, alpha)

        # Combine history and prediction to the series to be plotted
        history = np.array(self.history)
        prices = np.append(history, pred)

        # History days from the history series
        n_hist_period = len(self.history)
        pred_days = np.arange(n_period)
        days = np.arange(-n_hist_period, n_period)

        fig, ax = plt.subplots()
        ax.plot(days, prices)
        if plot_ci:
            ax.fill_between(pred_days, lower, upper, color='g', alpha=.1)
        plt.ylabel('Stock Prices')
        plt.xlabel('Days')
        plt.savefig(f"./plots/{self.params['SERIES_NAME']}_{self.name}_Price.png")
        plt.show()

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

    def simulate(self, dur, dt=1):
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

    def predict(self, dur):
        # Take the prediction path with least MSE to mean prediction
        preds = self.simulate(dur)
        mses = [np.average((preds[i] - preds.mean(axis=0)) ** 2) for i in range(preds.shape[0])]
        index = mses.index(min(mses))
        pred = preds[index]

        return pred

    def validate(self, test):
        dur = test.shape[0]
        preds = self.simulate(dur)
        mses = [np.average((preds[i] - test['Adj Close']) ** 2) for i in range(preds.shape[0])]
        mapes = [np.average((preds[i] - test['Adj Close']) ** 2 / preds[i])
                 for i in range(preds.shape[0])]
        return min(mapes), min(mses)

    def conf_int(self, n_period, alpha):
        preds = self.simulate(n_period)
        # Select the scenarios with (alpha)th quantile and (1-alpha)th quantile of stock price
        preds_mean = np.mean(preds, axis=0)
        preds_mean_sorted = np.sort(preds_mean)
        # Fetch the scenario number of (alpha)th quantile and (1-alpha)th quantile
        lower = preds_mean_sorted[round((len(preds_mean_sorted) - 1) * alpha)]
        upper = preds_mean_sorted[round((len(preds_mean_sorted) - 1) * (1 - alpha))]
        lower = np.where(preds_mean == lower)[0][0]
        upper = np.where(preds_mean == upper)[0][0]

        # Obtain the stock (alpha)th quantile and (1-alpha)th stock price series with the index obtained
        lower = preds[lower]
        upper = preds[upper]

        return lower, upper

    def summary(self):
        print("-" * 45)
        print(" " * 10 + "Geometric Brownian Motion" + " " * 10)
        print("-" * 45)

        print(f"The drift parameter mu is: {self.mu}")
        print(f"The diffusion parameter sigma is: {self.sigma}")
        return


class LassoForecaster(Forecaster):
    """
    Class for the Lasso Forecaster

    Attributes:
    raw_predictors: raw input predictors from the raw data, each of series of dimension n
    full_predictors: Horizontally stacked predictor matrix containing the time series
    transformed predictors 1, 2, ..., J
    """

    def __init__(self, params, name='Lasso'):
        super().__init__(params, name)
        # Stock price at the beginning of prediction
        self.s0 = 0
        self.lag = params['LAG']

        # Initialize the Lasso model
        # Grid Search parameters
        self.parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
        self.raw_predictors = None
        self.full_predictors = None
        self.n_predictors = 0
        self.model = Lasso()
        self.regressor = GridSearchCV(self.model, self.parameters, scoring='neg_mean_squared_error', cv=10)

    def fit(self, train):
        """
        Function fitting the model and set up the initial value of prediction
        Update historical stock price

        :param train: A dataframe of training stock price and its predictors
        :return: None
        """
        # Copy stock price info
        s = train['Adj Close']
        self.s0 = s.iloc[-1]

        # Drop the date to obtain predictors
        self.raw_predictors = train.drop(['Date'], axis=1)

        # Obtain differenced predictors
        raw_diff_predictors = self.raw_predictors.diff().shift(-1).iloc[:-1]

        # Fetch full predictor matrix with J predictors
        full_predictors = self.get_full_lasso_predictor(raw_diff_predictors)

        # Store time series predictors and number of predictors
        self.full_predictors = full_predictors
        self.n_predictors = self.raw_predictors.shape[1]

        # Obtain the differenced response vector
        # Truncate the differenced response to length n - lag to incorporate the predictors
        responses = train['Adj Close'].diff().shift(-1).iloc[:-1].values[self.lag:].reshape(-1, 1)
        # scale responses
        responses_scaled = (responses - responses.mean()) / responses.std()

        # Fit with the differenced and scaled response
        self.regressor.fit(full_predictors, responses_scaled)

    def get_full_lasso_predictor(self, raw_predictors):
        """
        Produce time series data for fitting the time series Lasso regression
        :return: predictors to be fitted for Lasso in numpy
        """
        p = raw_predictors.shape[1]
        predictors = None

        data = raw_predictors.values
        # Fore every time series predictor, create time series matrix
        for i in range(p):
            x = data[:, i]

            time_series_matrix = self._time_series_matrix(x, self.lag)

            # If processing the first predictor
            if i == 0:
                predictors = time_series_matrix
            else:
                # Stack the matrices horizontally
                predictors = np.hstack([predictors, self._time_series_matrix(x, self.lag)])

        return predictors

    @staticmethod
    def _time_series_matrix(x, q):
        """
        Get the time series matrix with lag q of a certain predictor
        :return: The time series (n-q) x q matrix with lag p with
        """
        n = x.shape[0]

        matrix = np.zeros([n - q, q])
        for i in range(q):
            matrix[:, i] = x[i:(n - q + i)]

        return matrix

    def validate(self, test):
        """
        :param test: A dataframe of testing stock price and its predictors
        :return: Testing MSE
        """
        dur = test.shape[0]
        responses = test['Adj Close'].values
        pred = self.predict(dur=dur)
        mse = np.average((pred - responses) ** 2)
        mape = np.average((pred - responses) ** 2 / pred)
        return mape, mse

    def predict(self, dur=0, raw_predictors=None):
        # If no raw predictors input, use the one from training
        if raw_predictors is None:
            raw_predictors = self.raw_predictors

        # Initialize previous prediction (Adjusted close)
        prev_pred = self.s0

        # Create differenced raw predictors
        raw_diff_predictors = self.raw_predictors.diff().shift(-1).iloc[:-1]
        # Create a copy of regressor
        regressor = self.regressor
        # Create prediction collection
        prediction = pd.Series([])

        for _ in range(dur):
            # Make prediction of (n-lag) vector
            full_predictors = self.get_full_lasso_predictor(raw_diff_predictors)
            pred_scaled = regressor.predict(full_predictors)  # Scaled predictor

            # Obtain stock price prediction
            pred = pred_scaled[-1] * raw_diff_predictors['Adj Close'].std() \
                   + raw_diff_predictors['Adj Close'].mean()

            # Scale back the predictors and add to last row as predictions
            raw_diff_predictors.loc[len(raw_diff_predictors)] = \
                (raw_diff_predictors.mean() + (pred_scaled[-1] * raw_diff_predictors.std())).values

            # Obtain the predicted stock price adding the difference to the previous stock price
            prev_pred = prev_pred + pred
            # Collect the predictions
            prediction = prediction.append(pd.Series(prev_pred))

            # Delete the first row and append
            raw_diff_predictors = raw_diff_predictors[1:].reset_index().iloc[:, 1:]

        return prediction

    def conf_int(self, n_period, alpha):
        # TODO: Finish CI bootstrapping
        """
        Function computing the confidence interval of the Lasso regression
        :param n_period: Number of period of prediction
        :param alpha: Significance level of confidence interval
        :return: None
        """
        # Create a copy of raw predictors from the input
        raw_predictors = self.raw_predictors
        # Collection of predictions over the scenarios
        predictions = pd.DataFrame()

        for _ in range(self.params['SCENE_SIZE']):
            # Bootstrapping sample from predictors and
            sample = raw_predictors.sample(frac=1, replace=True)
            pred = self.predict(dur=self.dur, raw_predictors=sample)
            # Append predictions to the collection
            predictions = predictions.append(dict(zip(predictions.columns, pred)), ignore_index=True)

        # Select the scenarios with (alpha)th quantile and (1-alpha)th quantile of stock price
        predictions_mean = np.mean(predictions, axis=0)
        predictions_mean_sorted = np.sort(predictions_mean)
        # Fetch the scenario number of (alpha)th quantile and (1-alpha)th quantile
        lower = predictions_mean_sorted[round((len(predictions_mean_sorted) - 1) * alpha)]
        upper = predictions_mean_sorted[round((len(predictions_mean_sorted) - 1) * (1 - alpha))]
        lower = np.where(predictions_mean == lower)[0][0]
        upper = np.where(predictions_mean == upper)[0][0]

        # Obtain the stock (alpha)th quantile and (1-alpha)th stock price series with the index obtained
        lower = predictions[lower]
        upper = predictions[upper]

        return lower, upper

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

    def plot_acf(self, series):
        """
        Show the plot of ACF
        @param series: The stock price series
        """
        plot_acf(series, lags=range(10))
        plt.savefig(f"./plots/{self.params['SERIES_NAME']}_ARIMA_ACF.png")
        plt.show()

    def plot_pacf(self, series):
        """
        Show the plot of PACF
        @param series: The stock price series
        """
        plot_pacf(series, lags=range(10))
        plt.savefig(f"./plots/{self.params['SERIES_NAME']}_ARIMA_PACF.png")
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
        n_period = test.shape[0]
        pred = self.predict(n_period)
        responses = test['Adj Close'].values.reshape(-1, 1)
        mse = np.average((pred - responses) ** 2)
        mape = np.average((pred - responses) ** 2 / responses)
        return mape, mse

    def conf_int(self, n_period, alpha=0.05):
        """
        Function producing the confidence interval of future stock price
        :param n_period: Number of period of prediction
        :param alpha: Significance level of confidence interval
        :return: None
        """
        _, conf_int = self.model.predict(n_period, return_conf_int=True, alpha=alpha)
        return conf_int[:, 0], conf_int[:, 1]

    def summary(self):
        print(self.model.summary())
