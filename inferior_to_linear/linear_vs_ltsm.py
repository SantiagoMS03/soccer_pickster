from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np

# Read CSV
premier_league_matches = pd.read_csv("./data/premier_league_matches.csv")

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

# LSTM Model
# Reshape data for LSTM input (samples, time steps, features)
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(2))  # Output layer with 2 units for HomeGoals and AwayGoals
model.compile(optimizer='adam', loss='mse')

# Train the LSTM model
model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)

# Make predictions on the testing data
y_pred_lstm = model.predict(X_test_lstm)

# Evaluate the model
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
print("LSTM Model Mean Squared Error:", mse_lstm)
