
import seaborn as sns
import pandas as pd
import numpy as np
import country_converter as coco
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

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

def give_sliding_window_volatility(df, window, volatility_variable):
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

def add_missing_variables(df, country):
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

    data_fci = pd.read_excel('./data/financial_condition_index.xlsx', sheet_name=None)
    data_fci['FCIs']['country'] = coco.convert(data_fci['FCIs']['country'], to='ISO2')
    data_cc_ew_fci = data_fci['FCIs'][data_fci['FCIs']['country'] == country][['date','country', 'ew_fci']]
    data_cc_ew_fci.rename(columns = {'country':'iso2', 'ew_fci':'financialConditionIndex'}, inplace = True)
    data_cc_ew_fci.index = pd.to_datetime(data_cc_ew_fci['date'])
    data_q_fci = pd.DataFrame()
    data_q_fci['financialConditionIndex'] = data_cc_ew_fci['financialConditionIndex'].resample('QS').mean()
    df['financialConditionIndex'] = data_q_fci['financialConditionIndex']
    df['policyRate'] = df['policyRate'].fillna(data_ea['policyRate'])
    df['EAtermspread'] = data_ea['EAtermSpread']
    df['USpolicyRate'] = data_us['policyRate']
    df['UStermSpread'] = data_us['UStermSpread']
    return df

def retrieved_processed_data(country_iso="DE", intervall="quarterly"):
    return pd.read_csv(f"./data_processed_{intervall}_{country_iso}.csv", index_col="date")

def get_xy_split(df, exclusion_x = ["is_systemic_crisis","month", "cpi_yoy_growthRate"], y_variable="is_systemic_crisis"):
    X_without = df.drop(exclusion_x, axis=1)
    y = df[y_variable]
    return X_without, y

def subselect_data(df, start_year = '1970'):
    df = df[df.index > start_year]
    df = df.dropna(axis=1)
    return df

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

def subselect_data(df, start_year = '1970'):
    df = df[df.index > start_year]
    df = df.dropna(axis=1)
    return df