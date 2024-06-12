
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
        df[growth_rate_variable+"_yoy"] = df[growth_rate_variable].pct_change(fill_method=None, periods=4)*100
    return df

def fx_volatility(df, window, volatility_variable):
    df[volatility_variable+"_std"] =  df[volatility_variable].rolling(window).std()*(window**0.5)
    return df

def add_systemic_risk_dummy(data_file, dummy_file, country):
    df = pd.read_csv(data_file)
    df['date'] = pd.PeriodIndex(df['date'], freq= 'Q').to_timestamp()

    dummy = pd.read_csv(dummy_file)
    dummy_de = dummy[dummy['iso2']==country]
    dummy_de['date'] = pd.to_datetime(dummy_de['date.1'])

    dummy_cc = dummy[dummy['iso2']==country]
    dummy_cc = dummy_cc.sort_values(by='date.1')
    dummy_cc.index = pd.to_datetime(dummy_cc['date.1'])

    df = pd.merge(df, dummy_de, on = ['iso2', 'date'], how= 'left')

    df = df.drop(['iso2','date.1_x','month','date.1_y','financialStressDummy'], axis=1)
    df['is_systemic_crisis'] = df['is_systemic_crisis'].fillna(0)
    df = df.dropna(axis=1, how = 'all')
    return df

def add_missing_variables(df):
    data_ea = pd.read_csv('./data/data_input_quarterly.csv')
    cols2add = ['date','policyRate', 'EAtermSpread']
    data_ea = data_ea[cols2add][data_ea['iso2']=='EA']
    data_ea['date'] = pd.PeriodIndex(data_ea['date'], freq= 'Q').to_timestamp()

    data_us = pd.read_csv('./data/data_input_quarterly.csv')
    cols2add = ['date','policyRate', 'UStermSpread']
    data_us = data_us[cols2add][data_us['iso2']=='US']
    data_us['date'] = pd.PeriodIndex(data_us['date'], freq= 'Q').to_timestamp()

    df.index = df['date']
    data_ea.index = data_ea['date']
    data_us.index = data_us['date']

    df['policyRate'] = df['policyRate'].fillna(data_ea['policyRate'])
    df['EAtermspread'] = data_ea['EAtermSpread']
    df['USpolicyRate'] = data_us['policyRate']
    df['UStermSpread'] = data_us['UStermSpread']

    df.index = df['date']
    df.drop('date', axis=1, inplace = True)
    return df


