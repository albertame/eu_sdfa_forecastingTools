

#%%
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

#%%
os.chdir(r'/Users/albertoamerico/Documents/eusdfa/git/eu_sdfa_forecastingTools')
from utils import read_data
from utils import calculate_growth_rates
from utils import get_lagged_variables
from utils import fx_volatility

# %%
file = './data/data_input_quarterly.csv'
country = 'DE'
df = read_data(file, country)
#%% Transform the data
yoy_variables = ["bankCreditPnfs", "totalCreditPnfsLCY", "totalCreditPnfs2GDP"]
df = calculate_growth_rates(df, yoy_variables)
df = df.drop(yoy_variables, axis=1)
lag2_variables = [f"{col}_yoy" for col in ["bankCreditPnfs", "totalCreditPnfsLCY", "totalCreditPnfs2GDP"]]
df = get_lagged_variables(df, 2, lag2_variables)
df = df.drop(lag2_variables, axis=1)
lag1_variables = [f"{col}_yoy" for col in ["cpi"]]
df = get_lagged_variables(df, 1, lag1_variables)
df = df.drop(lag1_variables, axis=1)
df.to_csv(f'data_quarterly_{country}.csv', index = False)

#%% 
data = pd.read_csv(f'./data_quarterly_{country}.csv')
data['date'] = pd.PeriodIndex(data['date'], freq= 'Q').to_timestamp()
data

dummy = pd.read_csv('dummy_final.csv')
dummy_de = dummy[dummy['iso2']==country]
dummy_de['date'] = pd.to_datetime(dummy_de['date.1'])
dummy_de

dummy_cc = dummy[dummy['iso2']==country]
dummy_cc = dummy_cc.sort_values(by='date.1')
dummy_cc.index = pd.to_datetime(dummy_cc['date.1'])
plt.plot(dummy_cc['is_systemic_crisis'])
plt.title(f'Systemic crisis dummy for {country}')

data = pd.merge(data, dummy_de, on = ['iso2', 'date'], how= 'left')
data.columns

data = data.drop(['iso2','date.1_x','month','date.1_y','financialStressDummy'], axis=1)
data['is_systemic_crisis'] = data['is_systemic_crisis'].fillna(0)
data = data.dropna(axis=1, how = 'all')

data_ea = pd.read_csv('./data/data_input_quarterly.csv')
cols2add = ['date','policyRate', 'EAtermSpread']
data_ea = data_ea[cols2add][data_ea['iso2']=='EA']
data_ea['date'] = pd.PeriodIndex(data_ea['date'], freq= 'Q').to_timestamp()
data_ea.dropna()

data_us = pd.read_csv('./data/data_input_quarterly.csv')
cols2add = ['date','policyRate', 'UStermSpread']
data_us = data_us[cols2add][data_us['iso2']=='US']
data_us['date'] = pd.PeriodIndex(data_us['date'], freq= 'Q').to_timestamp()
data_us.dropna()

data.index = data['date']
data_ea.index = data_ea['date']
data_us.index = data_us['date']

data['policyRate'] = data['policyRate'].fillna(data_ea['policyRate'])
data['EAtermspread'] = data_ea['EAtermSpread']
data['USpolicyRate'] = data_us['policyRate']
data['UStermSpread'] = data_us['UStermSpread']

data.index = data['date']
data.drop('date', axis=1, inplace = True)
data.columns

cols2reg = ['policyRate', 'resPropPrice', 'cpi_yoy', 'fx', 'financialStressIndex', 
'bankCreditPnfs_growthRate','totalCreditPnfsLCY_growthRate', 'totalCreditPnfs2GDP_growthRate',
 'fx_std', 'is_systemic_crisis', 'EAtermspread', 'USpolicyRate', 'UStermSpread']

for col in data.loc[data[data.index>'1970'].index, ~data.columns.isin(list(data.filter(regex = 'lag').columns))]:
plt.plot(data[col][data.index>'1970'])
plt.title([col])
plt.show()
# %%
