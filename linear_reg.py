import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.ensemble import BaggingRegressor

print(f"Starting the script")
print('-' * 50)
start_time = time.time()
curr_time = start_time
data = pd.read_csv('train.csv')

X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

y = np.log1p(y)

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

X_test_data = pd.read_csv('test.csv')

X_test_data[numeric_columns] = X_test_data[numeric_columns].fillna(X_test_data[numeric_columns].mean())

X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

datac = X_encoded.copy(deep=True)
datac['SalePrice'] = y
corr_matrix = datac.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
X_encoded.drop(columns=to_drop, inplace=True)

X_test_encoded = pd.get_dummies(X_test_data, columns=categorical_columns, drop_first=True)
X_test_encoded.drop(columns=to_drop, inplace=True)

for col in X_encoded.columns:
    if col not in X_test_encoded.columns:
        X_test_encoded[col] = 0

z_scores = np.abs(stats.zscore(X_encoded[numeric_columns]))
threshold = 10
mask = (z_scores < threshold).all(axis=1)
X_encoded = X_encoded[mask]
y = y[mask]

X_test_encoded = X_test_encoded.reindex(sorted(X_test_encoded.columns), axis=1)
X_encoded = X_encoded.reindex(sorted(X_encoded.columns), axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

selector = SelectKBest(score_func=f_regression, k=120)
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

print(f"Dataset preprocessed. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (Training): {mse_train}")
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error (Test): {mse_test}")

y_pred_test_data = model.predict(X_test_selected)
y_pred_test_original = np.exp(y_pred_test_data)
test_predictions = pd.DataFrame({'Id': X_test_data['Id'], 'SalePrice': y_pred_test_original})
test_predictions.to_csv('predictions_linreg.csv', index=False)
print(f"Linear regression model trained. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_model = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=0, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
best_rf_params = rf_grid_search.best_params_
print("Best Random Forest Hyperparameters:")
print(best_rf_params)
best_rf_model = RandomForestRegressor(random_state=42, **best_rf_params)
best_rf_model.fit(X_train, y_train)
y_pred_test_rf = best_rf_model.predict(X_test)
mse_test_rf = mean_squared_error(y_test, y_pred_test_rf)
print(f"Mean Squared Error (Random Forest - Test): {mse_test_rf}")
print(f"Random Forest model trained. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)
gb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gb_model = GradientBoostingRegressor(random_state=42)
gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=0, n_jobs=-1)
gb_grid_search.fit(X_train, y_train)
best_gb_params = gb_grid_search.best_params_
print("Best Gradient Boosting Hyperparameters:")
print(best_gb_params)
best_gb_model = GradientBoostingRegressor(random_state=42, **best_gb_params)
best_gb_model.fit(X_train, y_train)
y_pred_test_gb = best_gb_model.predict(X_test)
mse_test_gb = mean_squared_error(y_test, y_pred_test_gb)
print(f"Mean Squared Error (Gradient Boosting - Test): {mse_test_gb}")
print(f"Gradient Boosting Regressor trained. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 7],
    'min_child_weight': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=0, n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
best_xgb_params = xgb_grid_search.best_params_
print("Best XGBoost Hyperparameters:")
print(best_xgb_params)
best_xgb_model = xgb.XGBRegressor(random_state=42, **best_xgb_params)
best_xgb_model.fit(X_train, y_train)
y_pred_test_xgb = best_xgb_model.predict(X_test)
mse_test_xgb = mean_squared_error(y_test, y_pred_test_xgb)
print(f"Mean Squared Error (XGBoost - Test): {mse_test_xgb}")
print(f"XGBoost Regressor trained. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)
elastic_net = ElasticNet()
param_grid = {
    'max_iter': [1000, 1500, 2000],
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0],
    'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    'fit_intercept': [True, False],
    'positive': [True, False],
    'selection': ['cyclic', 'random']
}
elastic_net_model_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
elastic_net_model_search.fit(X_train, y_train)
best_enm_params = elastic_net_model_search.best_params_
print("Best ElasticNet HyperParameters:")
print(best_enm_params)
best_enm_model = ElasticNet(random_state=42, **best_enm_params)
best_enm_model.fit(X_train, y_train)
y_pred_test_enm = best_enm_model.predict(X_test)
mse_test_enm = mean_squared_error(y_test, y_pred_test_enm)
print(f"Mean Squared Error (Elastic Net - Test): {mse_test_enm}")
print(f"Elastic Net model trained. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)
catboost_param_grid = {
    'iterations': [500, 1000, 2000],
    'depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
}
catboost_model = CatBoostRegressor(random_seed=42, verbose=0)
catboost_grid_search = GridSearchCV(estimator=catboost_model, param_grid=catboost_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=0, n_jobs=-1)
catboost_grid_search.fit(X_train, y_train)
best_catboost_params = catboost_grid_search.best_params_
print("Best CatBoost Hyperparameters:")
print(best_catboost_params)
best_catboost_model = CatBoostRegressor(random_seed=42, verbose=0, **best_catboost_params)
best_catboost_model.fit(X_train, y_train)
y_pred_test_catboost = best_catboost_model.predict(X_test)
mse_test_catboost = mean_squared_error(y_test, y_pred_test_catboost)
print(f"Mean Squared Error (Test - CatBoost): {mse_test_catboost}")
print(f"CatBoost model trained. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)
y_pred_test_ensemble = (y_pred_test_rf + y_pred_test_gb + y_pred_test_xgb + y_pred_test_enm + y_pred_test_catboost) / 5
mse_test_ensemble = mean_squared_error(y_test, y_pred_test_ensemble)
print(f"Mean Squared Error (Ensemble - Test): {mse_test_ensemble}")
y_pred_test_data_rf = best_rf_model.predict(X_test_selected)
y_pred_test_data_gb = best_gb_model.predict(X_test_selected)
y_pred_test_data_xgb = best_xgb_model.predict(X_test_selected)
y_pred_test_data_enm = best_enm_model.predict(X_test_selected)
y_pred_test_data_catboost = best_catboost_model.predict(X_test_selected)
y_pred_test_data_ensemble = (y_pred_test_data_rf + y_pred_test_data_gb + y_pred_test_data_xgb + y_pred_test_data_enm + y_pred_test_data_catboost) / 5
y_pred_test_original_ensemble = np.exp(y_pred_test_data_ensemble)
test_predictions_ensemble = pd.DataFrame({'Id': X_test_data['Id'], 'SalePrice': y_pred_test_original_ensemble})
test_predictions_ensemble.to_csv('predictions_ensemble_5.csv', index=False)
print(f"Averaging ensemble completed. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)
epsilon = 1e-6
weights = {
    'random_forest': 1 / 10*(mse_test_rf + epsilon),
    'gradient_boosting': 1 / (mse_test_gb + epsilon),
    'xgboost': 1 / (mse_test_xgb + epsilon),
    'elastic_net': 1 / (mse_test_enm + epsilon),
    'catboost': 1 / (mse_test_catboost + epsilon)
}
total_weight = sum(weights.values())
normalized_weights = {model: weight / total_weight for model, weight in weights.items()}
y_pred_test_ensemble_weighted = (
    normalized_weights['random_forest'] * y_pred_test_rf +
    normalized_weights['gradient_boosting'] * y_pred_test_gb +
    normalized_weights['xgboost'] * y_pred_test_xgb +
    normalized_weights['elastic_net'] * y_pred_test_enm +
    normalized_weights['catboost'] * y_pred_test_catboost 
)
mse_test_ensemble_weighted = mean_squared_error(y_test, y_pred_test_ensemble_weighted)
print(f"Mean Squared Error (Weighted Ensemble - Test): {mse_test_ensemble_weighted}")
y_pred_test_data_ensemble_weighted = (
    normalized_weights['random_forest'] * y_pred_test_data_rf +
    normalized_weights['gradient_boosting'] * y_pred_test_data_gb +
    normalized_weights['xgboost'] * y_pred_test_data_xgb +
    normalized_weights['elastic_net'] * y_pred_test_data_enm +
    normalized_weights['catboost'] * y_pred_test_data_catboost 
)
y_pred_test_original_ensemble_weighted = np.exp(y_pred_test_data_ensemble_weighted)
test_predictions_ensemble_weighted = pd.DataFrame({'Id': X_test_data['Id'], 'SalePrice': y_pred_test_original_ensemble_weighted})
test_predictions_ensemble_weighted.to_csv('predictions_ensemble_5_weighted.csv', index=False)
print(f"Weighted averaging ensemble completed. Time: {abs(curr_time-((curr_time:=time.time()))):.2f}s")
print('-' * 50)