# PY Kaggle House Prices
 Kaggle Advanced Regression competition

## Methods

Testing different model ensembles on the Kaggle House Prices dataset - house prices prediction

### Results

1. Linear Regression
    - RMSE = 0.15305
2. Ensemble (Random Forest + Gradient Boosting Regressors)
    - RMSE = 0.14472
3. Ensemble (Random Forest + Gradient Boosting + XGBoost Regressors)
    - RMSE = 0.14449
4. Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost Regressors)
    - RMSE = 0.14014
5. Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors)
   - RMSE = 0.137
6. Weighted Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors)
   - RMSE = 0.13685
7. Weighted Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors + CatBoost)
   - RMSE = 0.13602
8. Weighted Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors + CatBoost) with feature selection
   - RMSE = 0.13341
9. Weighted Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors + CatBoost) with feature selection and elimination using z-score
   - **RMSE = 0.12845**
10. Hypertuned Neural Network with L1 Reg and Dropout layers
    - RMSE = 0.15791
11. Meta model - Linear Regression (5 stack - Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors + CatBoost)
    - RMSE = 0.17397
   
