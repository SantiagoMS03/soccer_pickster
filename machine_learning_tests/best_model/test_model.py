import pandas as pd
import joblib

# Load the trained model from the joblib file
model = joblib.load("linear_regressor.joblib")

# Read the new match data from the CSV file
new_match_data = pd.read_csv("match.csv")

# Read the feature names from the text file
with open('features.txt', 'r') as file:
    # Read the contents of the file and split by lines
    all_categories = file.read().splitlines()

# Create a DataFrame with all categories as columns
all_categories_df = pd.DataFrame(columns=all_categories)

# Merge the new match data with the DataFrame containing all categories
merged_data = pd.concat([new_match_data, all_categories_df], axis=1)

# Fill missing values with False (assuming missing features imply False)
merged_data.fillna(False, inplace=True)

# Use the trained model to make predictions on the new match data
predictions = model.predict(merged_data)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predictions, columns=['Predicted_HomeGoals', 'Predicted_AwayGoals'])

# Optionally, if you want to save the predictions to a new CSV file
predictions_df.to_csv("predictions.csv", index=False)

# Print the predictions
print(predictions_df)
