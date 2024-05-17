import pandas as pd

# Read the premier_league_matches CSV
premier_league_matches = pd.read_csv("premier_league_matches.csv")

# Select only the specified fields
selected_fields = premier_league_matches[['Season_End_Year', 'Home', 'Away', 'HomeGoals', 'AwayGoals']]

# Save the selected fields to a new CSV file
selected_fields.to_csv("relevant_features.csv", index=False)

print("CSV file created successfully.")
