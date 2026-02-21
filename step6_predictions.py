"""
Step 6: Predict home team win probability for all 16 matchups.
Uses logistic function and clips probabilities to [0.01, 0.99].
Outputs data/win_probabilities.csv
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

def main():
    power_df = pd.read_csv('data/power_rankings.csv')
    game_df = pd.read_csv('data/game_level_step1.csv')
    matchups_df = pd.read_csv('data/WHSDSC_Rnd1_matchups.csv')

    # Calculate empirical home win rate and home_adv
    emp_home_win_rate = game_df['home_win'].mean()
    home_adv_logit = np.log(emp_home_win_rate / (1 - emp_home_win_rate))
    print(f"Empirical home win rate: {emp_home_win_rate:.4f} -> home_adv: {home_adv_logit:.4f}")

    # Prepare rating lookup
    # Using xg_diff_per_game as rating
    rating_dict = dict(zip(power_df['team'], power_df['xg_diff_per_game']))

    # Prepare training data to find k
    game_df['home_rating'] = game_df['home_team'].map(rating_dict)
    game_df['away_rating'] = game_df['away_team'].map(rating_dict)
    rating_diffs = game_df['home_rating'] - game_df['away_rating']
    y_true = game_df['home_win'].values

    # Find best k by minimizing log loss
    def log_loss_fn(k, X, y, h_adv):
        logits = k * X + h_adv
        p = 1 / (1 + np.exp(-logits))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(log_loss_fn, args=(rating_diffs.values, y_true, home_adv_logit))
    k_opt = res.x
    print(f"Optimal scale factor (k): {k_opt:.4f}")

    # Predict matchups
    matchups_df['home_rating'] = matchups_df['home_team'].map(rating_dict)
    matchups_df['away_rating'] = matchups_df['away_team'].map(rating_dict)
    
    matchups_df['rating_diff'] = matchups_df['home_rating'] - matchups_df['away_rating']

    # p = sigmoid(k * rating_diff + home_adv)
    logits = k_opt * matchups_df['rating_diff'] + home_adv_logit
    p = 1 / (1 + np.exp(-logits))
    
    # Clip probabilities
    matchups_df['home_win_prob'] = np.clip(p, 0.01, 0.99)

    out_cols = ['game', 'game_id', 'home_team', 'away_team', 'home_win_prob']
    out_df = matchups_df[out_cols]
    
    out_path = 'data/win_probabilities.csv'
    out_df.to_csv(out_path, index=False)
    print(f"Step 6 complete. Saved {out_path} with {len(out_df)} predictions.")

if __name__ == "__main__":
    main()
