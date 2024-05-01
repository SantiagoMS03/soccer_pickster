from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import pandas as pd

# Read CSV
premier_league_matches = pd.read_csv("./data/premier_league_matches.csv")

# Split data into training and validation sets
train_size = int(len(premier_league_matches) * 0.8)
train, valid = premier_league_matches[:train_size], premier_league_matches[train_size:]

# Define the range of hyperparameters to search over
p_values = range(0, 3)  # Autoregressive terms
d_values = range(0, 3)  # Order of differencing
q_values = range(0, 3)  # Moving average terms

best_mse_home = float('inf')
best_params_home = None
best_mse_away = float('inf')
best_params_away = None

print("start1!")

# Perform hyperparameter tuning for HomeGoals
for p in p_values:
    print(p)
    for d in d_values:
        print(" ", d)
        for q in q_values:
            print("  ", q)
            try:
                # Fit ARIMA model for HomeGoals
                model_home = auto_arima(train['HomeGoals'], start_p=p, d=d, start_q=q, max_p=5, max_d=5, max_q=5, seasonal=False)
                
                # Make predictions for HomeGoals
                preds_home = model_home.predict(len(valid))
                
                # Calculate Mean Squared Error for HomeGoals
                mse_home = mean_squared_error(valid['HomeGoals'], preds_home)
                
                # Update best parameters if MSE is lower
                if mse_home < best_mse_home:
                    best_mse_home = mse_home
                    best_params_home = (p, d, q)
                    
            except:
                continue

print("done1!")

# Perform hyperparameter tuning for AwayGoals
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                # Fit ARIMA model for AwayGoals
                model_away = auto_arima(train['AwayGoals'], start_p=p, d=d, start_q=q, max_p=5, max_d=5, max_q=5, seasonal=False)
                
                # Make predictions for AwayGoals
                preds_away = model_away.predict(len(valid))
                
                # Calculate Mean Squared Error for AwayGoals
                mse_away = mean_squared_error(valid['AwayGoals'], preds_away)
                
                # Update best parameters if MSE is lower
                if mse_away < best_mse_away:
                    best_mse_away = mse_away
                    best_params_away = (p, d, q)
                    
            except:
                continue

print("Best Hyperparameters for HomeGoals:", best_params_home)
print("Best Mean Squared Error for HomeGoals:", best_mse_home)

print("Best Hyperparameters for AwayGoals:", best_params_away)
print("Best Mean Squared Error for AwayGoals:", best_mse_away)
