

#%%
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

#%%
#os.chdir(r'/Users/albertoamerico/Documents/eusdfa/git/eu_sdfa_forecastingTools')
from utils import read_data
from utils import calculate_growth_rates
from utils import get_lagged_variables
from utils import give_sliding_window_volatility

# %%
COUNTRY = 'DE'
FILE = './data/data_input_quarterly.csv'
TIME_INTERVALL = "quarterly"

def translate_frequency(intervall="quarterly"):
    if intervall == "quarterly":
        return "Q"

def get_processed_df(df, country="DE", time_intervall="quarterly", generate_graphs=False):
    
    yoy_variables = ["bankCreditPnfs", "totalCreditPnfsLCY", "totalCreditPnfs2GDP"]
    df = calculate_growth_rates(df, yoy_variables)
    df = df.drop(yoy_variables, axis=1)
    lag2_variables = [f"{col}_yoy" for col in ["bankCreditPnfs", "totalCreditPnfsLCY", "totalCreditPnfs2GDP"]]
    df = get_lagged_variables(df, 2, lag2_variables)
    df = df.drop(lag2_variables, axis=1)
    lag1_variables = [f"{col}_yoy" for col in ["cpi"]]
    df = get_lagged_variables(df, 1, lag1_variables)
    df = df.drop(lag1_variables, axis=1)
    df = give_sliding_window_volatility(df, 4, "fx")
    df.to_csv(f'data_{time_intervall}_{country}.csv', index = False)

    df['date'] = pd.PeriodIndex(df['date'], freq= translate_frequency(time_intervall)).to_timestamp()

    dummy = pd.read_csv('dummy_final.csv')
    dummy_of_country = dummy[dummy['iso2']==country]
    dummy_of_country['date'] = pd.to_datetime(dummy_of_country['date.1'])
    dummy_of_country = dummy_of_country.sort_values(by='date.1')
    dummy_of_country.index = pd.to_datetime(dummy_of_country['date.1'])

    df = pd.merge(df, dummy_of_country, on = ['iso2', 'date'], how= 'left')

    df = df.drop(['iso2','date.1_x','month','date.1_y','financialStressDummy'], axis=1)
    df['is_systemic_crisis'] = df['is_systemic_crisis'].fillna(0)
    df = df.dropna(axis=1, how = 'all')

    data_ea = pd.read_csv(f'./data/data_input_{time_intervall}.csv')
    cols2add = ['date','policyRate', 'EAtermSpread']
    data_ea = data_ea[cols2add][data_ea['iso2']=='EA']
    data_ea['date'] = pd.PeriodIndex(data_ea['date'], freq= translate_frequency(time_intervall)).to_timestamp()
    data_ea.dropna()

    data_us = pd.read_csv('./data/data_input_quarterly.csv')
    cols2add = ['date','policyRate', 'UStermSpread']
    data_us = data_us[cols2add][data_us['iso2']=='US']
    data_us['date'] = pd.PeriodIndex(data_us['date'], freq= 'Q').to_timestamp()
    data_us.dropna()

    df.index = df['date']
    data_ea.index = data_ea['date']
    data_us.index = data_us['date']

    df['policyRate'] = df['policyRate'].fillna(data_ea['policyRate'])
    df['EAtermspread'] = data_ea['EAtermSpread']
    df['USpolicyRate'] = data_us['policyRate']
    df['UStermSpread'] = data_us['UStermSpread']

    df.index = df['date']
    df.drop('date', axis=1, inplace = True)

    if generate_graphs:
        cols2reg = ['policyRate', 'resPropPrice', 'cpi_yoy', 'fx', 'financialStressIndex', 
        'bankCreditPnfs_growthRate','totalCreditPnfsLCY_growthRate', 'totalCreditPnfs2GDP_growthRate',
        'fx_std', 'is_systemic_crisis', 'EAtermspread', 'USpolicyRate', 'UStermSpread']

        for col in df.loc[df[df.index>'1970'].index, ~df.columns.isin(list(df.filter(regex = 'lag').columns))]:
            plt.plot(df[col][df.index>'1970'])
            plt.title([col])
            plt.show()

    return df


# %%
df = read_data(FILE, COUNTRY)
df = get_processed_df(df, COUNTRY,TIME_INTERVALL)
df.to_csv(f'data_processed_{TIME_INTERVALL}_{COUNTRY}.csv')
