# PY Kaggle House Prices
 Kaggle Advanced Regression competition

## Methods

Testing different model ensembles on the Kaggle House Prices dataset - house prices prediction

### Results

1. Linear Regression
    - RMSE = 0.15305
3. Ensemble (Random Forest + Gradient Boosting Regressors)
    - RMSE = 0.14472
4. Ensemble (Random Forest + Gradient Boosting + XGBoost Regressors)
    - RMSE = 0.14449
5. Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost Regressors)
    - RMSE = 0.14014
6. Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors)
   - RMSE = 0.137
7. Weighted Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors)
   - RMSE = 0.13685
9. Weighted Ensemble of hypertuned models (Random Forest + Gradient Boosting + XGBoost + ElasticNet Regressors + CatBoost)
   - RMSE = 0.13602
8. Hypertuned Neural Network with L1 Reg and Dropout layers
   - RMSE = 0.15791


