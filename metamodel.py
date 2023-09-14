import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from mlxtend.regressor import StackingCVRegressor


def blending(base_models, meta_model, X, y, X_test, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    meta_predictions = np.zeros((X.shape[0], len(base_models)))
    
    for i, model in enumerate(base_models):
        for train_idx, val_idx in kf.split(X):
            train_data, val_data = X[train_idx], X[val_idx]
            train_target, val_target = y[train_idx], y[val_idx]
            
            model.fit(train_data, train_target)
            val_pred = model.predict(val_data)
            meta_predictions[val_idx, i] = val_pred
    
    meta_model.fit(meta_predictions, y)
    
    test_predictions = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, model in enumerate(base_models):
        test_pred = model.predict(X_test)
        test_predictions[:, i] = test_pred
    
    ensemble_pred = meta_model.predict(test_predictions)
    
    return ensemble_pred


# Load the training dataset
data = pd.read_csv('train.csv')

X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

y = np.log1p(y)  # Log transformation

categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

X_test_data = pd.read_csv('test.csv')

X_test_data[numeric_columns] = X_test_data[numeric_columns].fillna(X_test_data[numeric_columns].mean())

X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_data, columns=categorical_columns, drop_first=True)

for col in X_encoded.columns:
    if col not in X_test_encoded.columns:
        X_test_encoded[col] = 0  # Include missing columns

X_test_encoded = X_test_encoded.reindex(sorted(X_test_encoded.columns), axis=1)
X_encoded = X_encoded.reindex(sorted(X_encoded.columns), axis=1)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Feature selection
selector = SelectKBest(score_func=f_regression, k=120)  # k for the number of features included
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)

# Split the set
X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Base Models

# RandomForest
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, scoring='neg_mean_squared_error',
                               cv=5, verbose=2, n_jobs=-1)
rf_grid_search.fit(X_train_base, y_train_base)
best_rf_params = rf_grid_search.best_params_
best_rf_model = RandomForestRegressor(random_state=42, **best_rf_params)
best_rf_model.fit(X_train_base, y_train_base)
y_pred_rf_base = best_rf_model.predict(X_train_meta)

# Gradient Boosting
gb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_model = GradientBoostingRegressor(random_state=42)
gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, scoring='neg_mean_squared_error',
                               cv=5, verbose=2, n_jobs=-1)
gb_grid_search.fit(X_train_base, y_train_base)
best_gb_params = gb_grid_search.best_params_
best_gb_model = GradientBoostingRegressor(random_state=42, **best_gb_params)
best_gb_model.fit(X_train_base, y_train_base)
y_pred_gb_base = best_gb_model.predict(X_train_meta)

# XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42)
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, scoring='neg_mean_squared_error',
                                cv=5, verbose=2, n_jobs=-1)
xgb_grid_search.fit(X_train_base, y_train_base)
best_xgb_params = xgb_grid_search.best_params_
best_xgb_model = xgb.XGBRegressor(random_state=42, **best_xgb_params)
best_xgb_model.fit(X_train_base, y_train_base)
y_pred_xgb_base = best_xgb_model.predict(X_train_meta)

# Elastic Net
elastic_net = ElasticNet()
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
elastic_net_model_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
elastic_net_model_search.fit(X_train_base, y_train_base)
best_enm_params = elastic_net_model_search.best_params_
best_enm_model = ElasticNet(random_state=42, **best_enm_params)
best_enm_model.fit(X_train_base, y_train_base)
y_pred_enm_base = best_enm_model.predict(X_train_meta)

# CatBoost
catboost_param_grid = {
    'iterations': [100, 200, 500, 1000],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
}

catboost_model = CatBoostRegressor(random_seed=42, verbose=0)
catboost_grid_search = GridSearchCV(estimator=catboost_model, param_grid=catboost_param_grid,
                                     scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)
catboost_grid_search.fit(X_train_base, y_train_base)
best_catboost_params = catboost_grid_search.best_params_
best_catboost_model = CatBoostRegressor(random_seed=42, verbose=0, **best_catboost_params)
best_catboost_model.fit(X_train_base, y_train_base)
y_pred_catboost_base = best_catboost_model.predict(X_train_meta)

# Stacking

meta_rf_model = RandomForestRegressor(random_state=42)
meta_rf_param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees
    'max_depth': [None, 10, 20],    # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Use GridSearchCV to find the best hyperparameters for the meta-model
meta_rf_grid_search = GridSearchCV(
    estimator=meta_rf_model,
    param_grid=meta_rf_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    n_jobs=-1
)
meta_rf_grid_search.fit(X_train_meta, y_train_meta)

# Get the best hyperparameters for the meta-model
best_meta_rf_params = meta_rf_grid_search.best_params_

# Create the Random Forest meta-model with the best hyperparameters
best_meta_rf_model = RandomForestRegressor(random_state=42, **best_meta_rf_params)
best_meta_rf_model.fit(X_train_meta, y_train_meta)

# StackingCVRegressor with the tuned Random Forest as the meta-model
stacked_model = StackingCVRegressor(
    regressors=[best_rf_model, best_gb_model, best_xgb_model, best_enm_model, best_catboost_model],
    meta_regressor=best_meta_rf_model,  # Use the tuned Random Forest as the meta-model
    cv=5,
    use_features_in_secondary=True
)
stacked_model.fit(X_train_meta, y_train_meta)

# Evaluate Stacked Model
y_pred_stacked = stacked_model.predict(X_test_selected)

# Convert predictions back to original scale
y_pred_test_original_stacked = np.exp(y_pred_stacked)

# Create a DataFrame for test predictions
test_predictions_stacked = pd.DataFrame({'Id': X_test_data['Id'], 'SalePrice': y_pred_test_original_stacked})

# Save the test predictions to a CSV file
test_predictions_stacked.to_csv('predictions_stacked_5_RF.csv', index=False)