"""
Step 5: Calculate power rankings for all 32 teams based on xG differential per game.
Calculates win percentage and merges finishing skill.
Outputs data/power_rankings.csv
"""
import pandas as pd
import numpy as np
import os

def main():
    game_df = pd.read_csv('data/game_level_step1.csv')
    fin_df = pd.read_csv('data/finishing_table_step4.csv')

    # Home stats
    home_stats = game_df.groupby('home_team').agg(
        games=('game_id', 'count'),
        xg_for=('home_xg_total', 'sum'),
        xg_against=('away_xg_total', 'sum'),
        wins=('home_win', 'sum')
    ).reset_index().rename(columns={'home_team': 'team'})

    # Away stats
    away_stats = game_df.groupby('away_team').agg(
        games=('game_id', 'count'),
        xg_for=('away_xg_total', 'sum'),
        xg_against=('home_xg_total', 'sum'),
        wins=('home_win', lambda x: (1-x).sum())
    ).reset_index().rename(columns={'away_team': 'team'})

    # Combine
    combined = pd.concat([home_stats, away_stats]).groupby('team').sum().reset_index()

    # Per game metrics
    combined['xg_diff_per_game'] = (combined['xg_for'] - combined['xg_against']) / combined['games']
    combined['win_percentage'] = combined['wins'] / combined['games']

    # Merge finishing
    merged = pd.merge(combined, fin_df, on='team', how='left')

    # Sort and rank
    merged = merged.sort_values('xg_diff_per_game', ascending=False).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)

    # Save
    out_path = 'data/power_rankings.csv'
    merged.to_csv(out_path, index=False)
    print(f"Step 5 complete. Saved {out_path} with {len(merged)} teams.")

if __name__ == "__main__":
    main()
