import pickle
import xarray as xr
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
import nc_time_axis
import xgboost as xgb
import tensorflow as tf
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# Load the netCDF datasets
ocean_current_data_vo_surface = xr.open_dataset('_data/_ocean_current_data_vo_surface.nc')
ocean_current_data_uo_surface = xr.open_dataset('_data/_ocean_current_data_uo_surface.nc')
temperature_data = xr.open_dataset('_data/_temperature_data.nc')
psl_data = xr.open_dataset('_data/_psl_data.nc')

# Prepare the data for training
avg_ocean_current_data_vo_surface = ocean_current_data_vo_surface.mean(dim=('lat', 'lon'))
avg_ocean_current_data_uo_surface = ocean_current_data_uo_surface.mean(dim=('lat', 'lon'))
avg_psl_data = psl_data.mean(dim=('lat', 'lon'))
avg_temperature_data = temperature_data.mean(dim=('lat', 'lon'))

mean_avg_ocean_current_data_vo_surface = avg_ocean_current_data_vo_surface.mean(dim='time')
sd_avg_ocean_current_data_vo_surface = avg_ocean_current_data_vo_surface.std(dim='time')

mean_avg_ocean_current_data_uo_surface = avg_ocean_current_data_uo_surface.mean(dim='time')
sd_avg_ocean_current_data_uo_surface = avg_ocean_current_data_uo_surface.std(dim='time')

mean_avg_psl_data = avg_psl_data.mean(dim='time')
sd_avg_psl_data = avg_psl_data.std(dim='time')

mean_avg_temperature_data = avg_temperature_data.mean(dim='time')
sd_avg_temperature_data = avg_temperature_data.std(dim='time')

avg_ocean_current_data_vo_surface_normalized = (avg_ocean_current_data_vo_surface - mean_avg_ocean_current_data_vo_surface) / sd_avg_ocean_current_data_vo_surface
avg_ocean_current_data_uo_surface_normalized = (avg_ocean_current_data_uo_surface - mean_avg_ocean_current_data_uo_surface) / sd_avg_ocean_current_data_uo_surface
avg_psl_data_normalized = (avg_psl_data - mean_avg_psl_data) / sd_avg_psl_data
avg_temperature_data_normalized = (avg_temperature_data - mean_avg_temperature_data) / sd_avg_temperature_data

X = pd.concat([avg_ocean_current_data_vo_surface_normalized['vo'].to_dataframe(), avg_ocean_current_data_uo_surface_normalized['uo'].to_dataframe(), avg_psl_data_normalized['psl'].to_dataframe()], axis=1)
X = X.apply(pd.to_numeric).fillna(0)

y = avg_temperature_data_normalized['tos']

# Create lag features
def create_lag_features(df, lags):
    lagged_df = pd.concat([df.shift(i) for i in range(lags + 1)], axis=1)
    lagged_df.columns = [f'{col}_lag{i}' for i in range(lags + 1) for col in df.columns]
    return lagged_df.dropna()

lags = 12  # Increase the number of lag features
X_lagged = create_lag_features(X, lags)
y_lagged = y[lags:]

# Split the data into training and testing sets
train_size = int(len(X_lagged) * 0.8)
X_train, X_test = X_lagged.iloc[:train_size], X_lagged.iloc[train_size:]
y_train, y_test = y_lagged[:train_size], y_lagged[train_size:]

# Global variables: Mean Squared Error
mse_rf = 0
mse_mlp = 0
mse_xgb = 0
mse_tf = 0

# RANDOM FOREST REGRESSOR

def randomforest():
    # Train the random forest regressor model
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mse_rf = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (Random Forest):", mse_rf)

    # Visualize the predicted values and the actual values
    sorted_indices = y_test.indexes['time'].argsort()
    sorted_y_test = y_test.values[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]+1
    plt.plot(sorted_y_test, label='Actual')
    plt.plot(sorted_y_pred, label='Predicted (Random Forest)')
    plt.xlabel('Time')
    plt.ylabel('Normalized Temperature')
    plt.legend()
    plt.show()

# Run the random forest function
randomforest()



# MLP REGRESSOR

def mlp():
    # Step 9: Train the MLP regressor model
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }
    mlp_model = GridSearchCV(MLPRegressor(max_iter=100), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    mlp_model.fit(X_train, y_train)

    # Step 10: Make predictions on the test set
    y_pred_mlp = mlp_model.predict(X_test)

    # Step 11: Evaluate the model
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    print("Mean Squared Error (MLP):", mse_mlp)

    # Step 12: Visualize the predicted values and the actual values for MLP model
    sorted_indices = y_test.indexes['time'].argsort()
    sorted_y_test = y_test.values[sorted_indices]
    sorted_y_pred_mlp = y_pred_mlp[sorted_indices]
    plt.plot(sorted_y_test, label='Actual')
    plt.plot(sorted_y_pred_mlp, label='Predicted (MLP)')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Temperature')
    plt.legend()
    plt.show()

# XGBoost REGRESSOR

def xgboost():
    # Step 13: Train the XGBoost regressor model
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=4,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.3,
        reg_lambda=0.8
    )
    xgb_model.fit(X_train.values, y_train.values)
    y_pred_xgb = xgb_model.predict(X_test.values)

    # Step 14: Make predictions on the test set
    y_pred_xgb = xgb_model.predict(X_test.values)

    # Step 15: Evaluate the model
    mse_xgb = mean_squared_error(y_test.values, y_pred_xgb)
    print("Mean Squared Error (XGBoost):", mse_xgb)

    # Step 16: Visualize the predicted values and the actual values for XGBoost model
    sorted_indices = y_test.indexes['time'].argsort()
    sorted_y_test = y_test.values[sorted_indices]
    sorted_y_pred_xgb = y_pred_xgb[sorted_indices]
    plt.plot(sorted_y_test, label='Actual')
    plt.plot(sorted_y_pred_xgb, label='Predicted (XGBoost)')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Temperature')
    plt.legend()
    plt.show()

# TENSORFLOW REGRESSOR

def tensorflow():
    # Step 17: Convert the data to TensorFlow tensors
    X_train_tf = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

    # Step 18: Define the TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_tf, y_train_tf, epochs=50, batch_size=32, verbose=0)
    mse_tf = model.evaluate(X_test_tf, y_test_tf)

    # Step 19: Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 20: Train the model
    model.fit(X_train_tf, y_train_tf, epochs=10, batch_size=32, verbose=0)

    # Step 21: Evaluate the model
    mse_tf = model.evaluate(X_test_tf, y_test_tf)
    print("Mean Squared Error (Deep Neural Network):", mse_tf)

    # Step 22: Make predictions on the test set
    y_pred_tf = model.predict(X_test_tf)

    # Step 23: Visualize the predicted values and the actual values for TensorFlow model
    sorted_indices = y_test.indexes['time'].argsort()
    sorted_y_test = y_test.values[sorted_indices]
    sorted_y_pred_tf = y_pred_tf[sorted_indices]
    plt.plot(sorted_y_test, label='Actual')
    plt.plot(sorted_y_pred_tf, label='Predicted (Deep Neural Network)')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Temperature')
    plt.legend()
    plt.show()


    # TENSORFLOW REGRESSOR WITH SEQUENTIAL DATA
    def tensorflow_sequential():
        # Step 17: Convert the data to TensorFlow tensors
        X_train_tf = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
        y_train_tf = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
        X_test_tf = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
        y_test_tf = tf.convert_to_tensor(y_test.values, dtype=tf.float32)
        
        # Step 18: Reshape the input data to include previous time steps
        n_steps = 3  # Number of previous time steps to consider
        X_train_seq = []
        y_train_seq = []
        X_test_seq = []
        y_test_seq = []
        
        for i in range(n_steps, len(X_train_tf)):
            X_train_seq.append(X_train_tf[i-n_steps:i])
            y_train_seq.append(y_train_tf[i])
        for i in range(n_steps, len(X_test_tf)):
            X_test_seq.append(X_test_tf[i-n_steps:i])
            y_test_seq.append(y_test_tf[i])
        
        X_train_seq = tf.stack(X_train_seq)
        y_train_seq = tf.stack(y_train_seq)
        X_test_seq = tf.stack(X_test_seq)
        y_test_seq = tf.stack(y_test_seq)
        
        # Step 19: Define the TensorFlow model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, activation='relu', input_shape=(n_steps, X_train.shape[1])),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Step 20: Train the model
        model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
        
        # Step 21: Evaluate the model
        mse_tf = model.evaluate(X_test_seq, y_test_seq)
        print("Mean Squared Error (Deep Neural Network with Sequential Data):", mse_tf)
        
        # Step 22: Make predictions on the test set
        y_pred_tf = model.predict(X_test_seq)
        
        # Step 23: Visualize the predicted values and the actual values for TensorFlow model
        sorted_indices = y_test.indexes['time'].argsort()[n_steps:]
        sorted_y_test = y_test.values[sorted_indices]
        sorted_y_pred_tf = y_pred_tf.flatten()
        
        plt.plot(sorted_y_test, label='Actual')
        plt.plot(sorted_y_pred_tf, label='Predicted (Deep Neural Network with Sequential Data)')
        plt.xlabel('Sample')
        plt.ylabel('Normalized Temperature')
        plt.legend()
        plt.show()

    tensorflow_sequential()

randomforest()
mlp()
xgboost()
tensorflow()

# RESULTS

# Step 24: Compare the performance of the models

def result():
    models = ['Random Forest', 'MLP', 'XGBoost', 'TensorFlow']
    mse_scores = [mse_rf, mse_mlp, mse_xgb, mse_tf]

    best_model_index = mse_scores.index(min(mse_scores))
    best_model = models[best_model_index]
    best_mse = mse_scores[best_model_index]

    print("Best Model:", best_model)

result()