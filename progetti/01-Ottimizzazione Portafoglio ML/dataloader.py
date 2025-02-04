import numpy as np
import pandas as pd
from fredapi import Fred

from typing import Tuple,List
import yaml

def __read_config(config_file:str = 'config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_fed_data(ticker:str = 'WTB3MS', start_date:str = '2000-01-01', end_date:str = '2021-01-01') -> pd.DataFrame:

    config = __read_config()
    API_KEY = config['API_KEY']['FRED_KEY']
    fred = Fred(API_KEY)

    return fred.get_series(ticker, start_date, end_date)