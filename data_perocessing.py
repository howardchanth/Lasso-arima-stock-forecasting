import numpy as np


class DataProcessor:
    def __init__(self, raw_data, params):
        self.raw_data = raw_data
        self.params = params
        self.n_observations = raw_data.shape[0]
        self.n_predictors = raw_data.shape[1]
        self.lag = self.params['LAG']

    def subset_by_dates(self):

        return self.raw_data

    def get_lasso_predictor(self):
        """
        Produce time series data for fitting the time series Lasso regression
        :return: predictors to be fitted for Lasso in numpy
        """
        n = self.n_observations
        p = self.n_predictors
        q = self.lag

        data = self.raw_data.to_numpy()
        # Fore every time series predictor, create time series matrix
        predictors = self._time_series_matrix(data[:, 0], q)
        for i in range(1, p):
            x = data[:, 0]
            # Stack the matrices horizontally
            predictors = np.hstack([predictors, self._time_series_matrix(x, q)])

        return predictors

    def _time_series_matrix(self, x, q):
        """
        Get the time series matrix with lag q of a certain predictor
        :return: The time series (n-q) x q matrix with lag p with
        """
        n = self.n_observations
        matrix = np.zeros([n - q, q])
        for i in range(q):
            matrix[:, i] = x[i:(n - q + i)]

        return matrix
