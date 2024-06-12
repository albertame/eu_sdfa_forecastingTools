#%% Load functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from utils import retrieved_processed_data
from utils import get_xy_split

#%% Retrieving processing data
df = retrieved_processed_data("DE", "quarterly")

# %% Split data
Y_VARIABLE = "is_systemic_crisis"
X_EXCLUSION = ["is_systemic_crisis","month"]
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