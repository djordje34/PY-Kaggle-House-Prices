import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from kerastuner.tuners import Hyperband  

def build_model(hp):
    model = Sequential()
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(Dense(
            units=hp.Int('units', min_value=8, max_value=512, step=8),
            activation=hp.Choice('activation', values=['relu']),
            input_shape=(X_train.shape[1],),
            kernel_regularizer=regularizers.l2(hp.Float('regularizer', min_value=1e-5, max_value=1e-2, sampling='log'))
        ))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    #model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='linear'))
        
    model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
            ),
            loss='mse',
            metrics=['mse']
        )
    return model

data = pd.read_csv('train.csv')

X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

#Log transformation
y = np.log1p(y)

#Onehot
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

X_test_data = pd.read_csv('test.csv')

numeric_columns_test = X_test_data.select_dtypes(include=[np.number]).columns.tolist()
X_test_data[numeric_columns_test] = X_test_data[numeric_columns_test].fillna(X_test_data[numeric_columns_test].mean())

categorical_columns_test = X_test_data.select_dtypes(include=['object']).columns.tolist()
X_test_encoded = pd.get_dummies(X_test_data, columns=categorical_columns_test, drop_first=True)

for col in X_encoded.columns:
    if col not in X_test_encoded.columns:
        X_test_encoded[col] = 0  #add missing cols

X_test_encoded = X_test_encoded.reindex(sorted(X_test_encoded.columns), axis=1)
X_encoded = X_encoded.reindex(sorted(X_encoded.columns), axis=1)

print([x for x in X_encoded.columns if x not in X_test_encoded.columns])

#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

#Select best features
selector = SelectKBest(score_func=f_regression, k=80)  #Adjust 'k' as needed
X_selected = selector.fit_transform(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)



tuner = Hyperband(
    build_model,  #Function to build the model
    objective='val_mse',  #Metric to optimize
    max_epochs=300, 
    factor=3, 
    directory='hyperband_logs',  
    project_name='house_prices_nn_logs'
)

# hyperband tuner
best_model_callback = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_mse',
    mode='min',
    save_best_only=True
)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mse', 
                                              patience=5
                                              )
# Perform the Hyperband search
tuner.search(
    X_train, y_train,
    epochs=300, 
    validation_data=(X_test, y_test),
    callbacks=[best_model_callback,stop_early]  #save best arch model
)
print([x.values for x in tuner.get_best_hyperparameters(num_trials=5)])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(best_hps.values)
#best_model = tuner.get_best_models(num_models=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test))

val_mse_per_epoch = history.history['val_mse']
best_epoch = val_mse_per_epoch.index(min(val_mse_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)
best_model.save('best_model.h5')
print(best_model.summary())

y_pred_test = best_model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error (Test): {mse_test}, RMSE: {np.sqrt(np.exp(mse_test))}")

#SHOULD HAVE MADE A FUNCTION FOR PREPROCESSING BUT KINDA LAZY TODAY

X_test_scaled = scaler.transform(X_test_encoded)

X_test_selected = selector.transform(X_test_scaled)

y_pred_test_data = best_model.predict(X_test_selected)

y_pred_test_original = np.exp(y_pred_test_data)
y_pred_test_original = y_pred_test_original.reshape(-1)
print(y_pred_test_original,np.shape(y_pred_test_original))
test_predictions = pd.DataFrame({'Id': X_test_data['Id'], 'SalePrice': y_pred_test_original})

test_predictions.to_csv('predictions_nn.csv', index=False)