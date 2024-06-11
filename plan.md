### SUPTECH forecasting tool
#### Step 1: explore the dataset
- Global liquidity indicators
- Residential Property Pries
- Consumer Prices
- Exchange rates
- Bank credit to PNFS
- Total credit (household, NFC, PNFS)
- Debt service ratio (household, NFC, PNFS)
- Policy rate
- EA 10-year government bond yield
- EA 2-year government bond yield
- EA Financial Stress Index
- US term spread (10y - 2y)
- Financial stress index

#### Step 2: define a target variable
- Systemic crisis dummy (binary variable)
- Financial stress index
- Inflation

#### Step 3: feature selection
#### Step 4: choose models for initial benchmarking 
- Random Forest
- XGBoost
- Ridge
- Neural Network
- OLS
- VAR
- Logistic Regression

#### Step 5: choose forecasting horizon
#### Step 5: choose evaluation metrics
- Accuracy score
- F1-score
- RMSE

#### Step 7: Hyperparameter tuning
#### Step 8: Integrate MLFLOW to store experiments results

#### Step 8: evaluate the modelsn peformance
#### Step 9: visualise the results for the end-user
