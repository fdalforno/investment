import numpy as np
import cvxpy as cp
import pandas as pd

class TrendFilter:
    def __init__(self, k: int = 1):
        # Initialize the TrendFilter with the order of the difference matrix
        self.k = k

    def __make_diff_matrix(self, n: int) -> np.ndarray:
        # Create the difference matrix for trend filtering
        diags = [i for i in range(1, self.k + 1) for _ in range(2)]
        Dfull = np.diag([self.k] * n)
        for i in range(self.k):
            j = -(-1) ** (i + 1) * (i // 2 + 1)
            Dfull -= np.diag([1] * (n - diags[i]), j)
        return Dfull[0:(n - diags[i]), ]

    def fit(self, data: pd.Series, lam: float = 1.0) -> pd.Series:
        # Fit the trend filter to the data
        X_val = data.values

        n = np.size(X_val)
        x_ret = X_val.reshape(n)

        # Create the difference matrix
        D = self.__make_diff_matrix(n)

        # Define the optimization variable and parameter
        beta = cp.Variable(n)
        lambd = cp.Parameter(nonneg=True)

        # Set the lambda value
        lambd.value = lam

        # Define the trend filtering objective function
        def tf_obj(x, beta, lambd):
            return cp.norm(x - beta, 2) ** 2 + lambd * cp.norm(cp.matmul(D, beta), 1)

        # Define and solve the optimization problem
        problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))
        problem.solve()

        # Return the fitted values as a pandas Series
        betas = pd.Series(beta.value, index=data.index)
        return betas