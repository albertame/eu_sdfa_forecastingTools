---
marp: true
class: invert
title: Forcasting and nowcasting
---

# Forcasting and nowcasting

Georgios, Nicholas, Robert, Francesco, Thomas, Alberto

---

# Plan

1. Data exploration
2. Data preperation
3. Feature manipulation
4. Model training

---

# Feature manipulation

```python
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


```
