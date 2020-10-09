from scipy import stats
from statsmodels.tsa.stattools import adfuller
import numpy as np


class RunsTester:
    """
    Input:
    series : A series of positive and negative values, in this case stock returns
    alpha  : The significance level of the test

    Output: The results of hypothesis testing

    """

    def __init__(self, series):
        self.series = series

        self.returns = ((self.series.loc[1:] - self.series.shift(1).loc[1:])
                        / self.series.shift(1).loc[1:])

        runs, n_positive, n_negative = 0, 0, 0
        # Get runs, n_pos and n_neg
        for i in range(2, len(self.returns) + 1):
            if (self.returns[i] > 0 > self.returns[i - 1]) or \
                    (self.returns[i] < 0 < self.returns[i - 1]):
                runs += 1

            if self.returns[i] > 0:
                n_positive += 1
            else:
                n_negative += 1

        # Initialize the tester
        self.runs = runs
        self.n_positive = n_positive

        self.n_negative = n_negative

    def get_z_statistic(self):
        # Calculate the test statistics
        runs_exp = (2 * self.n_negative * self.n_negative) / (self.n_negative + self.n_negative) + 1
        var_runs = ((runs_exp - 1) * (2 * self.n_negative * self.n_negative - self.n_negative - self.n_negative)
                    / (self.n_negative + self.n_negative) / (self.n_negative + self.n_negative - 1))
        sd_runs = np.sqrt(var_runs)
        z = (self.runs - runs_exp) / sd_runs

        return z

    def test(self, alpha):
        # Use the z statistics obtained to test the hypothesis
        # H0: The sequence is produced in a random manner
        # H1: The sequence is not produced in a random manner
        z = self.get_z_statistic()
        p_value = stats.norm.sf(abs(z))
        print("-" * 15 + "Runs Test" + "-" * 15)
        print(f"The Z statistic is: {z}")
        print(f"The p value is {p_value}")
        if p_value > alpha:
            print("Does not reject H0 - The sequence may be produced in a random manner")
        else:
            print("Reject H0 - The sequence is not produced in a random manner")

        return


class StationarityTester:
    # Dicky-Fuller
    def __init__(self, series, params):
        self.series = series
        self.params = params

    def test(self):
        alpha = self.params['ALPHA']
        adf_test = adfuller(self.series)
        p_value = adf_test[1]
        print("-" * 15 + "Dicky-Fuller Test" + "-" * 15)
        print(f"The value of test statistics: is {adf_test[0]}")
        print(f"The p value is: {adf_test[1]}")
        if p_value > alpha:
            print("Does not reject H0 - The stock price time series is not stationary")
            print("ARIMA can be effectively applied on the stock price series")
        else:
            print("Reject H0 - The stock price time series is stationary\n")
        return adf_test

    def get_d(self):
        alpha = self.params['ALPHA']
        series = self.series
        for d in range(1, 6):
            series = series.diff().shift(-1).iloc[:-1]
            adf_test = adfuller(series)
            p_value = adf_test[1]
            if p_value < alpha:
                return d
