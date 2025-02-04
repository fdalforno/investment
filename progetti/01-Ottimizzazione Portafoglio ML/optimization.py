from typing import List, Dict, Union
import cvxpy as cp
import numpy as np
import pandas as pd
from math import inf

eps = 1e-10

class Constraint:
 
    def generate_constraint(self, variables: Dict):
        """ Create the cvxpy Constraint
 
        :param variables: dictionary containing the cvxpy Variables for the
          problem
        :return: A cvxpy Constraint object representing the constraint
        """
        pass

class LongOnlyConstraint(Constraint):
    
    def __init__(self):
        """ Constraint to enforce all portfolio weights are non-negative
        """
        pass
 
    def generate_constraint(self, variables: Dict):
        return variables['w'] >= 0
 
 
class FullInvestmentConstraint(Constraint):
 
    def __init__(self):
        """ Constraint to enforce the sum of the portfolio weights is one
        """
        pass
 
    def generate_constraint(self, variables: Dict):
        return cp.sum(variables['w']) == 1.0

class TrackingErrorConstraint(Constraint):
 
    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 reference_weights: pd.Series,
                 sigma: pd.DataFrame,
                 upper_bound: float):
        """ Constraint on the tracking error between a subset of the
        portfolio and a set of target weights
 
        :param asset_names: Names of all assets in the problem
        :param reference_weights: Vector of target weights. Index should be
          a subset of asset_names
        :param sigma: Covariance matrix, indexed by asset_names
        :param upper_bound: Upper bound for the constraint, in units of
          volatility (standard deviation)
        """
        self.reference_weights = \
            reference_weights.reindex(asset_names).fillna(0)
        self.sigma = sigma
        self.upper_bound = upper_bound ** 2
 
    def generate_constraint(self, variables: Dict):
        w = variables['w']
        tv = cp.quad_form(w - self.reference_weights, self.sigma)
        return tv <= self.upper_bound

class VolatilityConstraint(TrackingErrorConstraint):
 
    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 sigma: pd.DataFrame,
                 upper_bound: float):
        """ Constraint on the overall volatility of the portfolio
 
        :param asset_names: Names of all assets in the problem
        :param sigma: Covariance matrix, indexed by asset_names
        :param upper_bound: Upper bound for the constraint, in units of
          volatility (standard deviation)
        """
 
        zeros = pd.Series(np.zeros(len(asset_names)), asset_names)
        super(VolatilityConstraint, self).__init__(asset_names, zeros,
                                                   sigma, upper_bound)

class ReturnConstraint(Constraint):
    def __init__(self,
                 returns: pd.Series,
                 lower_bound: float):
        """ Constraint on the expected return of the portfolio
        
        :param returns: Expected returns for each asset
        :param lower_bound: Lower bound for the expected return of the
          portfolio
        """
        
        self.returns = returns
        self.lower_bound = lower_bound
    
    def generate_constraint(self, variables: Dict):
        w = variables['w']
        mu = self.returns
        return w.T @ mu >= self.lower_bound

class MeanVarianceOpt:
 
    def __init__(self):
        self.asset_names = []
        self.variables = None
        self.prob = None
        self.has_solution = False
        self.solution = inf
 
    @staticmethod
    def _generate_constraints(variables: Dict,
                              constraints: List[Constraint]):
        return [c.generate_constraint(variables) for c in constraints]
 
    def solve(self):
        result = self.prob.solve()
        status = self.prob.status
        
        if status not in ["infeasible", "unbounded"]:
            print("Solution {0:.2} result status {1}".format(result,status))
            self.has_solution = True
            self.solution = result
        else:
            print("WARNING:  the optimizer did NOT exit successfully!!")
            
 
    def get_var(self, var_name: str):
        return pd.Series(self.variables[var_name].value, self.asset_names)

class MaxExpectedReturnOpt(MeanVarianceOpt):

    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 constraints: List[Constraint],
                 ers: pd.Series):
        super().__init__()
        self.asset_names = asset_names
        variables = dict({'w': cp.Variable(len(ers))})

        cons = MeanVarianceOpt._generate_constraints(variables,
                                                     constraints)
        obj = cp.Maximize(ers.values.T @ variables['w'])
        self.variables = variables
        self.prob = cp.Problem(obj, cons)

class MinVarianceOpt(MeanVarianceOpt):
    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 constraints: List[Constraint],
                 sigma: pd.DataFrame):
        
        super().__init__()
        self.asset_names = asset_names
        self.sigma = sigma
        variables = dict({'w': cp.Variable(len(asset_names))})

        cons = MeanVarianceOpt._generate_constraints(variables,
                                                     constraints)
        
        obj = cp.Minimize(cp.quad_form(variables['w'],self.sigma))
        self.variables = variables
        self.prob = cp.Problem(obj, cons)


class RiskParityOpt(MeanVarianceOpt):
    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 constraints: List[Constraint],
                 sigma: pd.DataFrame,
                 gamma: float):
        
        super().__init__()
        self.asset_names = asset_names
        
        variables = dict({'w': cp.Variable(len(asset_names))})

        cons = MeanVarianceOpt._generate_constraints(variables,
                                                     constraints)
        
        obj = cp.Minimize(0.5 * cp.quad_form(variables['w'],sigma) - gamma * sum(cp.log(variables['w'])))
        self.variables = variables
        self.prob = cp.Problem(obj, cons)


#https://yetanothermathprogrammingconsultant.blogspot.com/2016/08/portfolio-optimization-maximize-sharpe.html


class MaxSharpeRatioOpt(MeanVarianceOpt):
    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 ers: pd.Series,
                 sigma: pd.DataFrame,
                 rf_rate: float):
        
        super().__init__()
        self.asset_names = asset_names
        
        variables = dict({'w': cp.Variable(len(ers)),'t':cp.Variable()})
        
        cons = []

        mu_hat = ers - rf_rate
        cons.append(mu_hat.values.T @ variables['w'] == 1)
        cons.append(variables['t'] >= 0)
        cons.append(LongOnlyConstraint().generate_constraint(variables))
        cons.append(cp.sum(variables['w']) == variables['t'])
        
        obj = cp.Minimize(cp.quad_form(variables['w'],sigma))

        self.variables = variables
        self.prob = cp.Problem(obj, cons)

    
