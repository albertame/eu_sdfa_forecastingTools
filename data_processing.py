#%%
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

#%%
os.chdir(r'/Users/albertoamerico/Documents/eusdfa/git/eu_sdfa_forecastingTools')
from utils import read_data

# %%
read_data('./data/data_input_quarterly.csv','DE')
GROWTH_VARIABLES = ["cpi_yoy", "bankCreditPnfs", "totalCreditPnfsLCY", "totalCreditPnfs2GDP"]
df = calculate_growth_rates(df, GROWTH_VARIABLES)

# %%
