import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
global_temp = pd.read_csv('GlobalTemperatures.csv')

# Handle missing values
global_temp = global_temp.dropna()

# Feature Engineering
global_temp['Year'] = pd.to_datetime(global_temp['dt']).dt.year

# Normalize the data
scaler = StandardScaler()
global_temp[['LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']] = scaler.fit_transform(
    global_temp[['LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']]
)

# Train-Test Split
X = global_temp[['Year', 'LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']]
y = global_temp['LandAndOceanAverageTemperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Use np.sqrt to calculate RMSE
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, RMSE: {rmse}, R2: {r2}')

# Load the state-level dataset
state_temp = pd.read_csv('GlobalLandTemperaturesByState.csv')

# Handle missing values
state_temp = state_temp.dropna()

# Feature Engineering
state_temp['Year'] = pd.to_datetime(state_temp['dt']).dt.year

# Normalize the data
state_temp[['AverageTemperature']] = scaler.fit_transform(state_temp[['AverageTemperature']])

# Define the RandomForestRegressor model
rf = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
print("Best parameters:", grid_search.best_params_)

# Plot actual vs predicted temperatures
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('Temperature')
plt.title('Actual vs Predicted Temperatures')

# Ensure the plot is displayed
plt.show(block=True)  # Use block=True to force the plot display
