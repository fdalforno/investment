import numpy as np
import pandas as pd

def annualize_rets(r:pd.DataFrame, periods_per_year:int)-> float:
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r:pd.Series, periods_per_year:int)-> float:
    return r.std()*(periods_per_year**0.5)

def geometric_mean(r:pd.Series,periods_per_year:int)-> float:
    n_periods = r.shape[0]
    return np.exp(np.sum(np.log(1+r)) * periods_per_year/n_periods) - 1

def annualize_sharpe(r:pd.DataFrame, risk_free_rate:pd.DataFrame, periods_per_year:int)-> float:
    n_periods = r.shape[0]
    
    ret_expected = np.sum(r-risk_free_rate)/n_periods
    ret_avg = np.sum(r)/n_periods

    std_dev = np.sqrt(np.sum((r - ret_avg)**2 ) / n_periods)
    annu_ret_expected = (ret_expected+1)**n_periods-1
    annu_std_dev = std_dev * (periods_per_year**0.5)

    return annu_ret_expected/annu_std_dev


def est_return(w:pd.Series,mu:pd.Series)-> float:
    return w @ mu


def est_vol(w:pd.Series,sigma:pd.DataFrame)-> float:
    return (w.T @ sigma @ w)**0.5

def est_sharpe(w:pd.Series,rf:float,mu:pd.Series,sigma:pd.DataFrame) -> float:
    return (est_return(w,mu)-rf)/est_vol(w,sigma)


def max_drawdown(r:pd.DataFrame)-> float:
    wealth_index = 1000*(1+r).cumprod()
    cummax = wealth_index.cummax()
    drawdown = (wealth_index - cummax)/cummax
    return drawdown.min()