#%% Load functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def retrieved_processed_data(country_iso="DE", intervall="quarterly"):
    return pd.read_csv(f"./data_{intervall}_{country_iso}.csv")

def get_xy_split(df, exclusion_x = ["is_systemic_crisis","month", "cpi_yoy_growthRate"], y_variable="is_systemic_crisis"):
    X_without = df.drop(exclusion_x, axis=1)
    y = df[y_variable]
    return X_without, y
#%% Retrieving processing data
df = retrieved_processed_data("DE", "quarterly")

# %% Split data
Y_VARIABLE = "is_systemic_crisis"
X_EXCLUSION = ["is_systemic_crisis","month", "cpi_yoy_growthRate"]
x, y = get_xy_split(df, X_EXCLUSION, Y_VARIABLE)

# %% Standardize data
X = StandardScaler().fit_transform(x)
# %% PCA - data
principalComponents = pca.fit_transform(X)

# %% Graph PCA
pca = PCA()
principalComponents = pca.fit_transform(X)

features_length = 12
features = range(1,features_length)
plt.bar(features, pca.explained_variance_ratio_[:features_length-1], color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.figure(figsize=(20, 2))

PCA_components = pd.DataFrame(principalComponents)

# %% Graph PC
df.columns