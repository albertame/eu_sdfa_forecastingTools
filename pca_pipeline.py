#%% Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from utils import read_data
from utils import get_processed_df
from utils import subselect_data
from utils import get_xy_split

COUNTRY = 'DE'
FILE = './data/data_input_quarterly.csv'
TIME_INTERVALL = "quarterly"
df = read_data(FILE, COUNTRY)
df = get_processed_df(df, COUNTRY,TIME_INTERVALL, verbose=True)
df = subselect_data(df)

Y_VARIABLE = "is_systemic_crisis"
X_EXCLUSION = ["is_systemic_crisis"]

x_SPLIT, y_SPLIT = get_xy_split(df, X_EXCLUSION, Y_VARIABLE)
y_SPLIT = y_SPLIT.reset_index(drop=True)

X_SCALED = StandardScaler().fit_transform(x_SPLIT)

pca = PCA()
principalComponents = pca.fit_transform(X_SCALED)
PCA_components = pd.DataFrame(principalComponents)

explained_variance_ratio = pca.explained_variance_ratio_
target_variance = 0.80
current_variance = 0.0
num_features = 0

while current_variance < target_variance:
    current_variance += explained_variance_ratio[num_features]
    num_features += 1

print(f"The minimum number of PCA features needed to explain at least {target_variance} of the variance is: {num_features}")

FILTER_NUMBER = len(PCA_components)-80

X_TRAIN = PCA_components.iloc[:FILTER_NUMBER,:num_features]
X_GOLD = PCA_components.iloc[FILTER_NUMBER:,:num_features]
Y_TRAIN = y_SPLIT[:FILTER_NUMBER]
Y_GOLD = y_SPLIT[FILTER_NUMBER:]

clf = LogisticRegression(random_state=0).fit(X_TRAIN, Y_TRAIN)
predictions = clf.predict(X_GOLD)

plt.figure(figsize=(10, 5))
plt.plot(np.array(Y_GOLD), label='True Values', marker='o')
plt.plot(predictions, label='Predictions', marker='x')
plt.title('Logistic Regression Predictions vs True Values')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.show()
# %%
