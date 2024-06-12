import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def retrieved_processed_data(country_iso="DE", intervall="quaterly"):
    return pd.read_csv(f"./data_{intervall}_{country_iso}.csv")

def get_xy_split(df):
    return

# X_without = df_cleaned.drop(["is_systemic_crisis","month", "cpi_yoy_growthRate"], axis=1)

# X = StandardScaler().fit_transform(X_without)

# pca = PCA()
# principalComponents = pca.fit_transform(X)

# features_length = 12

# features = range(1,features_length)
# plt.bar(features, pca.explained_variance_ratio_[:features_length-1], color='black')
# plt.xlabel('PCA features')
# plt.ylabel('variance %')
# plt.xticks(features)
# plt.figure(figsize=(20, 2))

# PCA_components = pd.DataFrame(principalComponents)