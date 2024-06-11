
import seaborn as sns
import pandas as pd
import numpy as np

def read_data(file, country):
    df = pd.read_csv(file)
    df = df[df["iso2"] == country]
    return df

def get_lagged_variables(df, lags, variables):
    lag_array = np.arange(1,lags+1)
    for lag in lag_array:
        for variable in variables:
            df[variable+"_L"+str(lag)] = df[variable].shift(lag)
    return df

def calculate_growth_rates(df, growth_rate_variables):
    for growth_rate_variable in growth_rate_variables:
        df[growth_rate_variable+"_growthRate"] = df[growth_rate_variable].pct_change(fill_method=None, periods=4)*100
    return df

def fx_volatility(df, window, volatility_variable):
    df[volatility_variable+"_std"] =  df[volatility_variable].rolling(window).std()*(window**0.5)
    return df




