

#%% Imports
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from utils import read_data
from utils import calculate_growth_rates
from utils import get_lagged_variables
from utils import give_sliding_window_volatility
from utils import add_missing_variables

# %% Data processing function
def add_systemic_risk_dummy_with_df(df, dummy_df, country="DE"):
    df = df.reset_index(drop=True)
    print("subselect dummy in field")
    dummy_country = dummy_df[dummy_df['iso2']==country]
    dummy_country['date'] = pd.to_datetime(dummy_country['date.1'])
    df['date'] = pd.to_datetime(df['date'])
    print("found dummy country list")
    df = pd.merge(df, dummy_country, on=['iso2', 'date'], how= 'left')

    df = df.drop(['iso2','date.1_x','month','date.1_y','financialStressDummy'], axis=1)
    df['is_systemic_crisis'] = df['is_systemic_crisis'].fillna(0)
    df = df.dropna(axis=1, how = 'all')

    return df

def translate_frequency(intervall="quarterly"):
    if intervall == "quarterly":
        return "Q"

def get_processed_df(df, country="DE", time_intervall="quarterly", generate_graphs=False, verbose=False):
    df = give_sliding_window_volatility(df, 4, "fx")
    yoy_variables = ["bankCreditPnfs", "totalCreditPnfsLCY", "totalCreditPnfs2GDP"]
    df = calculate_growth_rates(df, yoy_variables)
    df = df.drop(yoy_variables, axis=1)
    
    lag2_variables = [f"{col}_yoy" for col in ["bankCreditPnfs", "totalCreditPnfsLCY", "totalCreditPnfs2GDP"]]
    df = get_lagged_variables(df, 2, lag2_variables)
    df = df.drop(lag2_variables, axis=1)
    lag1_variables = [f"{col}_yoy" for col in ["cpi"]]
    df = get_lagged_variables(df, 1, lag1_variables)
    df = df.drop(lag1_variables, axis=1)
    if verbose:
        print("successful added lag")

    df = add_missing_variables(df, country)
    if verbose:
        print("Added missing values")
    df['financialStressIndex_movingAverage'] = df['financialStressIndex'].rolling(12).mean()
    if verbose:
        print("Imputed moving average")
    
    df_dummies = pd.read_csv('dummy_final.csv')
    if verbose:
        print("Successful read of dummy .csv")
    df = add_systemic_risk_dummy_with_df(df, df_dummies, country)
    if verbose:
        print("Added systemic risk dummy")

    if generate_graphs:
        for col in df.loc[df[df.index>'1970'].index, ~df.columns.isin(list(df.filter(regex = 'lag').columns))]:
            plt.plot(df[col][df.index>'1970'])
            plt.title([col])
            plt.show()
    
    df.index = df['date']
    df.drop('date', axis=1, inplace = True)
    return df

# %% Run process
COUNTRY = 'DE'
FILE = './data/data_input_quarterly.csv'
TIME_INTERVALL = "quarterly"
df = read_data(FILE, COUNTRY)
df = get_processed_df(df, COUNTRY,TIME_INTERVALL, verbose=True)
# %%
df.to_csv(f'data_processed_{TIME_INTERVALL}_{COUNTRY}.csv', index=True)

# %%
df
