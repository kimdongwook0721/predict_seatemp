import xarray as xr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the netCDF datasets
ocean_current_data_vo_surface = xr.open_dataset('_data/_ocean_current_data_vo_surface.nc')
ocean_current_data_uo_surface = xr.open_dataset('_data/_ocean_current_data_uo_surface.nc')
temperature_data = xr.open_dataset('_data/_temperature_data.nc')
psl_data = xr.open_dataset('_data/_psl_data.nc')

# Step 3: Prepare the data for training and testing
# Combine the features into a single dataframe
features = pd.concat([ocean_current_data_vo_surface.to_dataframe().reset_index(drop=True), 
                      ocean_current_data_uo_surface.to_dataframe().reset_index(drop=True), 
                      psl_data.to_dataframe().reset_index(drop=True)], axis=1)
features = features[:768]  # Limit the number of samples to match the target variable

# Extract the target variable and limit to the same number of samples
target = temperature_data['tos'].isel(time=slice(0, 768))

# Normalize the features
scaler_X = StandardScaler()
features_normalized = scaler_X.fit_transform(features)

# Initialize arrays to store the combined results
predicted_temp_combined = np.zeros_like(target.values)
actual_temp_combined = target.values.copy()

# Create lagged features
def create_lagged_features(data, lags):
    lagged_data = data.copy()
    for lag in range(1, lags + 1):
        lagged_data = np.hstack((lagged_data, np.roll(data, shift=lag, axis=0)))
    return lagged_data[lags:, :], lags

lags = 3
features_lagged, lag = create_lagged_features(features_normalized, lags)
target_lagged = target.values[lags:]

# Split the data into training and testing sets for each lat*lon point and train models
rf_models = []
mse_list = []
time_splits = 5

for lat in range(target.shape[1]):
    for lon in range(target.shape[2]):
        y = target_lagged[:, lat, lon].reshape(-1, 1)
        scaler_y = StandardScaler()
        y_normalized = scaler_y.fit_transform(y).flatten()

        tscv = TimeSeriesSplit(n_splits=time_splits)
        
        for train_index, test_index in tscv.split(features_lagged):
            X_train, X_test = features_lagged[train_index], features_lagged[test_index]
            y_train, y_test = y_normalized[train_index], y_normalized[test_index]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)
            
            # Predict on the entire dataset for visualization purposes
            y_full_pred_normalized = model.predict(features_lagged)
            y_full_pred = scaler_y.inverse_transform(y_full_pred_normalized.reshape(-1, 1)).flatten()

            predicted_temp_combined[lags:, lat, lon] = y_full_pred

# Combine the results into a single dataset
combined_results = xr.Dataset(
    {
        "predicted_temperature": (("time", "lat", "lon"), predicted_temp_combined),
        "actual_temperature": (("time", "lat", "lon"), actual_temp_combined),
    },
    coords={
        "time": target.time.values,
        "lat": target.lat.values,
        "lon": target.lon.values,
    },
)

# Print Mean Squared Error for reference
print("Mean Squared Error List:", mse_list)

# Calculate and plot the average over the entire time period
average_predicted_temp = combined_results["predicted_temperature"].mean(dim="time")
average_actual_temp = combined_results["actual_temperature"].mean(dim="time")
average_difference = average_predicted_temp - average_actual_temp

print(average_difference)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
average_predicted_temp.plot(ax=axes[0], vmin=0, vmax=35, cmap='coolwarm')
average_actual_temp.plot(ax=axes[1], vmin=0, vmax=35, cmap='coolwarm')
average_difference.plot(ax=axes[2], cmap='coolwarm')
axes[0].set_title("Average Predicted Temperature")
axes[1].set_title("Average Actual Temperature")
axes[2].set_title("Average Difference (Predicted - Actual)")
plt.tight_layout()
plt.show()

# Visualize the results for each time period
for t in range(combined_results["time"].shape[0]):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
    combined_results["predicted_temperature"].isel(time=t).plot(ax=axes[0], vmin=0, vmax=35, cmap='coolwarm')
    combined_results["actual_temperature"].isel(time=t).plot(ax=axes[1], vmin=0, vmax=35, cmap='coolwarm')
    difference = combined_results["predicted_temperature"].isel(time=t) - combined_results["actual_temperature"].isel(time=t)
    difference.plot(ax=axes[2], vmin=-5, vmax=5, cmap='coolwarm')
    axes[0].set_title(f"Predicted Temperature at time={t}")
    axes[1].set_title(f"Actual Temperature at time={t}")
    axes[2].set_title(f"Difference (Predicted - Actual) at time={t}")
    plt.tight_layout()
    plt.show()