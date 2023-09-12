import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

#dataset
data = pd.read_csv('train.csv')

X = data.drop(columns=['SalePrice'])  
y = data['SalePrice']  

numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

y = np.log1p(y) #log transformation

#for onehot
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

X_test_data = pd.read_csv('test.csv')

X_test_data[numeric_columns] = X_test_data[numeric_columns].fillna(X_test_data[numeric_columns].mean())

#onehot encoding
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)


X_test_encoded = pd.get_dummies(X_test_data, columns=categorical_columns, drop_first=True)

for col in X_encoded.columns:
    if col not in X_test_encoded.columns:
        X_test_encoded[col] = 0  #Include missing cols

X_test_encoded = X_test_encoded.reindex(sorted(X_test_encoded.columns), axis=1)
X_encoded = X_encoded.reindex(sorted(X_encoded.columns), axis=1)
print([x for x in X_encoded.columns if x not in X_test_encoded.columns])
#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

#Feature selection
selector = SelectKBest(score_func=f_regression, k=80)  # k for the number of features included
X_selected = selector.fit_transform(X_scaled, y)
X_test_selected = selector.transform(X_test_scaled)

#Splitting the set
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42) #split for optimal testing without testing the original set

model = LinearRegression() #simple linear reg
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)

mse_train = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error (Training): {mse_train}")

y_pred_test = model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test) #calc MSE
print(f"Mean Squared Error (Test): {mse_test}")

#test predictions
y_pred_test_data = model.predict(X_test_selected)
y_pred_test_original = np.exp(y_pred_test_data)

test_predictions = pd.DataFrame({'Id': X_test_data['Id'], 'SalePrice': y_pred_test_original})
test_predictions.to_csv('predictions_linreg.csv', index=False)

rf_param_grid = {
    'n_estimators': [50, 100, 150],  #number of trees
    'max_depth': [None, 10, 20],  #Maximum depth of each tree
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#rand forest regressor
rf_model = RandomForestRegressor(random_state=42)

#grid search for optimal hyperparams
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

best_rf_params = rf_grid_search.best_params_
print("Best Random Forest Hyperparameters:")
print(best_rf_params)

#train model with best hyperparams
best_rf_model = RandomForestRegressor(random_state=42, **best_rf_params)
best_rf_model.fit(X_train, y_train)

y_pred_test_rf = best_rf_model.predict(X_test)

mse_test_rf = mean_squared_error(y_test, y_pred_test_rf)
print(f"Mean Squared Error (Random Forest - Test): {mse_test_rf}")


################
gb_param_grid = {
    'n_estimators': [50, 100, 150],  #Number of boosting stages to be used
    'learning_rate': [0.01, 0.1, 0.2],  #Step size shrinkage to prevent overfitting
    'max_depth': [3, 4, 5],  #Maximum depth of each tree
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#grad boost reg.
gb_model = GradientBoostingRegressor(random_state=42)

#same as in rand forest
gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)
gb_grid_search.fit(X_train, y_train)

best_gb_params = gb_grid_search.best_params_
print("Best Gradient Boosting Hyperparameters:")
print(best_gb_params)

best_gb_model = GradientBoostingRegressor(random_state=42, **best_gb_params)
best_gb_model.fit(X_train, y_train)

y_pred_test_gb = best_gb_model.predict(X_test)

mse_test_gb = mean_squared_error(y_test, y_pred_test_gb)
print(f"Mean Squared Error (Gradient Boosting - Test): {mse_test_gb}")


##########
xgb_param_grid = {
    'n_estimators': [50, 100, 150],  #Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  #Step size shrinkage to prevent overfitting
    'max_depth': [3, 4, 5],  #Maximum depth of each tree
    'min_child_weight': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],  #Fraction of samples used for training
    'colsample_bytree': [0.8, 0.9, 1.0]  #Fraction of features used for training
}

# xgb reg
xgb_model = xgb.XGBRegressor(random_state=42)

xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)

best_xgb_params = xgb_grid_search.best_params_
print("Best XGBoost Hyperparameters:")
print(best_xgb_params)

best_xgb_model = xgb.XGBRegressor(random_state=42, **best_xgb_params)
best_xgb_model.fit(X_train, y_train)

y_pred_test_xgb = best_xgb_model.predict(X_test)

mse_test_xgb = mean_squared_error(y_test, y_pred_test_xgb)
print(f"Mean Squared Error (XGBoost - Test): {mse_test_xgb}")


##########################
y_pred_test_ensemble = (y_pred_test_rf + y_pred_test_gb + y_pred_test_xgb) / 3

#mse for ensemble on train
mse_test_ensemble = mean_squared_error(y_test, y_pred_test_ensemble)
print(f"Mean Squared Error (Ensemble - Test): {mse_test_ensemble}")


y_pred_test_data_rf = best_rf_model.predict(X_test_selected)
y_pred_test_data_gb = best_gb_model.predict(X_test_selected)
y_pred_test_data_xgb = best_xgb_model.predict(X_test_selected)

#combine preds using averaging
y_pred_test_data_ensemble = (y_pred_test_data_rf + y_pred_test_data_gb + y_pred_test_data_xgb) / 3

#convert from log representation
y_pred_test_original_ensemble = np.exp(y_pred_test_data_ensemble)

test_predictions_ensemble = pd.DataFrame({'Id': X_test_data['Id'], 'SalePrice': y_pred_test_original_ensemble})

test_predictions_ensemble.to_csv('predictions_ensemble_3.csv', index=False)