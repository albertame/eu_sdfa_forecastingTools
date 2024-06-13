

#%% Imports
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from utils import read_data
from utils import get_processed_df

# %% Run process
COUNTRY = 'DE'
FILE = './data/data_input_quarterly.csv'
TIME_INTERVALL = "quarterly"
df = read_data(FILE, COUNTRY)
df = get_processed_df(df, COUNTRY,TIME_INTERVALL, verbose=True)
# %%
df.to_csv(f'data_processed_{TIME_INTERVALL}_{COUNTRY}.csv', index=True)
# %%
