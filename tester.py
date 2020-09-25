from scipy import stats
import numpy as np


class RunTester:
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

        print(f"The Z statistic is: {z}")
        print(f"The p value is {p_value}")
        if p_value > alpha:
            print("Does not reject H0 - The sequence may be produced in a random manner")
        else:
            print("Reject H0 - The sequence is not produced in a random manner")

        return


class StationaryTester:
    pass
