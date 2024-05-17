from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Read CSV
premier_league_matches = pd.read_csv("./data/premier_league_matches.csv")

# Perform one-hot encoding for 'Home' and 'Away'
premier_league_matches = pd.get_dummies(premier_league_matches, columns=['Home', 'Away'])

# Convert 'Season_End_Year' to numerical categories
premier_league_matches['Season_End_Year'] = pd.Categorical(premier_league_matches['Season_End_Year'])
premier_league_matches['Season_End_Year'] = premier_league_matches['Season_End_Year'].cat.codes

# Separate features (X) and target variable (y)
X = premier_league_matches.drop(columns=['HomeGoals', 'AwayGoals', 'Date', 'FTR', 'Wk', 'Season_End_Year'])
y = premier_league_matches[['HomeGoals', 'AwayGoals']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data for LSTM
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(2))  # Output layer with 2 neurons for HomeGoals and AwayGoals
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
