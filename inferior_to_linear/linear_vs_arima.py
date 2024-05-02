from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse

# Read CSV
premier_league_matches = pd.read_csv("../data/premier_league_matches.csv")

# Perform one-hot encoding for 'Home' and 'Away'
premier_league_matches = pd.get_dummies(premier_league_matches, columns=['Home', 'Away'])

# Separate features (X) and target variable (y)
X = premier_league_matches.drop(columns=['HomeGoals', 'AwayGoals', 'Date', 'FTR', 'Wk', 'Season_End_Year'])
y = premier_league_matches[['HomeGoals', 'AwayGoals']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(len(X_scaled))

# Filter training and testing data based on year
latest_year = premier_league_matches['Season_End_Year'].max()
train_indices = premier_league_matches['Season_End_Year'] != latest_year
test_indices = premier_league_matches['Season_End_Year'] == latest_year

# Print the number of records in each set
print("Number of training records:", train_indices.sum())
print("Number of testing records:", test_indices.sum())

print(train_indices)
print(test_indices)

# Filter data using boolean indexing
X_train = X_scaled[train_indices]
y_train = y[train_indices]
X_test = X_scaled[test_indices]
y_test = y[test_indices]

# Initialize a linear regression model
regressor = LinearRegression()

# Train the model on the filtered training data
regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Linear Regressor Mean Squared Error:", mse)

# ARIMA Model
# Convert the target variable to a univariate series
y_arima = y['HomeGoals']

# Split the dataset into training and testing sets
y_train_arima = y_arima[train_indices]
y_test_arima = y_arima[test_indices]

# Reset index for y_test_arima
y_test_arima.reset_index(drop=True, inplace=True)

# Fit ARIMA model
model_arima = ARIMA(y_train_arima, order=(5,1,0))  # Example order, you can optimize this
model_fit = model_arima.fit()

# Make predictions on the testing data
y_pred_arima = model_fit.forecast(steps=len(y_test_arima))

# Evaluate the model
mse_arima = mean_squared_error(y_test_arima, y_pred_arima)
print("ARIMA Model Mean Squared Error:", mse_arima)
