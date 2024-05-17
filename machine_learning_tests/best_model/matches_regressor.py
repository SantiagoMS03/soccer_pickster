from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

import pandas as pd

# Read CSV
premier_league_matches = pd.read_csv("relevant_features.csv")

# Perform one-hot encoding for 'Home' and 'Away'
premier_league_matches = pd.get_dummies(premier_league_matches, columns=['Home', 'Away'])

# Separate features (X) and target variable (y)
X = premier_league_matches.drop(columns=['HomeGoals', 'AwayGoals'])
y = premier_league_matches[['HomeGoals', 'AwayGoals']]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a linear regression model
regressor = LinearRegression()

# Train the model on the training data
regressor.fit(X_train, y_train)


# Make predictions on the testing data
y_pred = regressor.predict(X_test)

# Evaluate the modelt
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

joblib.dump(regressor, 'linear_regressor.joblib')

feature_names = X_train.columns

# Read the new match data from the CSV file
new_match_data = pd.read_csv("match.csv")

# Ensure that the new match data contains all the required features
missing_features = set(feature_names) - set(new_match_data.columns)
for feature in missing_features:
    new_match_data[feature] = False

# Fill missing values with False (assuming missing features imply False)
new_match_data.fillna(False, inplace=True)

# Select only the columns that were used during model training
new_match_data = new_match_data[X_train.columns]

# Use the trained model to make predictions on the new match data
predictions = regressor.predict(new_match_data)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predictions, columns=['Predicted_HomeGoals', 'Predicted_AwayGoals'])

# Optionally, if you want to save the predictions to a new CSV file
predictions_df.to_csv("predictions.csv", index=False)

# Print the predictions
print(predictions_df)

# NEEDS NORMALIZATION!