#%% Load functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import retrieved_processed_data
from utils import get_xy_split
from utils import subselect_data

#%% Retrieving processing data
df = retrieved_processed_data("DE", "quarterly")

#%% Retrieving processing data
df = subselect_data(df)

# %% Split data
Y_VARIABLE = "is_systemic_crisis"
X_EXCLUSION = ["is_systemic_crisis"]
x_SPLIT, y_SPLIT = get_xy_split(df, X_EXCLUSION, Y_VARIABLE)

# %% Standardize data
X_SCALED = StandardScaler().fit_transform(x_SPLIT)
# %% Graph PCA
pca = PCA()
principalComponents = pca.fit_transform(X_SCALED)
features_length = 7
features = range(1,features_length)
plt.bar(features, pca.explained_variance_ratio_[:features_length-1], color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.figure(figsize=(20, 2))

PCA_components = pd.DataFrame(principalComponents)
# %% Graph PCA
np.sum(pca.explained_variance_ratio_[:5])


# %% Graph PCA
from sklearn.linear_model import LogisticRegression
FILTER_NUMBER = len(PCA_components)-80
PCANUMBER = 5
X_TRAIN = PCA_components.iloc[:FILTER_NUMBER,:PCANUMBER]
X_GOLD = PCA_components.iloc[FILTER_NUMBER:,:PCANUMBER]
Y_TRAIN = y_SPLIT[:FILTER_NUMBER]
Y_GOLD = y_SPLIT[FILTER_NUMBER:]
# %% Graph PCA
clf = LogisticRegression(random_state=0).fit(X_TRAIN, Y_TRAIN)
# %% Graph PCA
# Making predictions
predictions = clf.predict(X_GOLD)

# Plotting the predictions against the true values
plt.figure(figsize=(10, 5))
plt.plot(Y_GOLD, label='True Values', marker='o')
plt.plot(predictions, label='Predictions', marker='x')
plt.title('Logistic Regression Predictions vs True Values')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.show()