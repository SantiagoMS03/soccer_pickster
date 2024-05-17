import pandas as pd

# Read CSV's
points_table = pd.read_csv("./data/points_table.csv")
premier_league_matches = pd.read_csv("./data/premier_league_matches.csv")

# Merge points_table with premier_league_matches based on Home and Away
merged_df = pd.merge(points_table, premier_league_matches, left_on='Team', right_on='Home', how='left')
merged_df = pd.merge(merged_df, premier_league_matches, left_on='Team', right_on='Away', how='left')

# Drop duplicate columns
merged_df.drop(['Home_y', 'Away_y'], axis=1, inplace=True)

# Export the merged DataFrame to a CSV file
merged_df.to_csv("merged_dataset_without_player_stats.csv", index=False)
