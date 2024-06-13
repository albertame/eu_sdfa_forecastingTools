---
marp: true
class: invert
title: Forcasting and nowcasting
---

# Forcasting and nowcasting

Alberto, Vittorio, Georgios, Nicholas, Robert, Francesco, Tiago, Thomas,

---

# Plan

1. Data exploration
2. Data preperation
3. Data pre-processing
4. Model training

---

# 1.1 Exploration

Descriptive statistics

---

# 2.1 Data preperation pipeline

```python
COUNTRY = 'DE'
FILE = './data/data_input_quarterly.csv'
TIME_INTERVALL = "quarterly"

df = read_data(FILE, COUNTRY)
df = get_processed_df(df, COUNTRY,TIME_INTERVALL, verbose=True)
df = subselect_data(df)

```

---

# 2.2 Data preperation pipeline

```python
df = give_sliding_window_volatility(df, 4, "fx")
df = calculate_growth_rates(df, yoy_variables)
df = get_lagged_variables(df, 2, lag2_variables)
df = add_missing_variables(df, country)
df = add_systemic_risk_dummy_with_df(df, df_dummies, country)

```

---

# 3.1 PCA intuition

- Principal component analysis (PCA) reduces the number of dimensions in large datasets to principal components
- Retain original information.
- Tansforming potentially correlated variables into a smaller set of variables, called principal components.

(between us: just let Sklearn do the magic)

---

# 3.2 PCA Pipeline

```python
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
```
