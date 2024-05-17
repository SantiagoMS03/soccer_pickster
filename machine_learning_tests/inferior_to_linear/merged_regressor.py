from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd

merged_df = pd.read_csv("merged_dataset_without_player_stats.csv")

print("Finished reading!")

# 1. Data Preprocessing
# Assuming 'merged_df' contains the merged dataset with relevant features and the target variable 'FinalScore'
X = merged_df[merged_df.columns]  # Select relevant features
y = merged_df[['HomeGoals_x', 'AwayGoals_x']] # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("after splitting")

# 2. Choose a Regressor
regressor = LinearRegression()  # Initialize a linear regression model

# 3. Train the Regressor
regressor.fit(X_train, y_train)  # Train the model on the training data

print("after training")

# 4. Predict Final Score
y_pred = regressor.predict(X_test)  # Make predictions on the testing data

# 5. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)  # Calculate mean squared error
print("Mean Squared Error:", mse)

# 6. Fine-Tuning (Optional)
# Optionally, you can fine-tune hyperparameters or try different regression algorithms
