"""
Step 7: Visualizing disparity ratio vs xG differential per game.
Produces DataScienceTeam.png.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def main():
    power_df = pd.read_csv('data/power_rankings.csv')
    disp_df = pd.read_csv('data/line_disparity_section2.csv')

    # Merge
    merged = pd.merge(power_df, disp_df[['team', 'disparity_ratio']], on='team', how='inner')

    # Assign colors based on rank
    def get_color(rank):
        if rank <= 10: return 'green'
        if rank <= 22: return 'gold'
        return 'red'

    merged['color'] = merged['rank'].apply(get_color)

    x = merged['disparity_ratio'].values
    y = merged['xg_diff_per_game'].values

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c=merged['color'], edgecolors='black', s=80, alpha=0.8)

    # Trendline
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_reg = slope * x_range + intercept
    
    ax.plot(x_range, y_reg, color='blue', linestyle='--', linewidth=2, 
            label=f"Regression line\nR² = {r_squared:.4f}")

    # Annotate tier legend
    import matplotlib.patches as mpatches
    g_patch = mpatches.Patch(color='green', label='Top 10')
    y_patch = mpatches.Patch(color='gold', label='Middle 12')
    r_patch = mpatches.Patch(color='red', label='Bottom 10')
    reg_patch = mpatches.Patch(color='blue', label='Regression')
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([g_patch, y_patch, r_patch])
    ax.legend(handles=handles)

    ax.set_title("Line Disparity Ratio vs xG Differential Per Game")
    ax.set_xlabel("Line Disparity Ratio (Line 1 xG/60 vs Line 2 xG/60)")
    ax.set_ylabel("xG Differential Per Game")
    
    # Save
    out_png = 'DataScienceTeam.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    
    print(f"Step 7 complete. Saved {out_png}")

if __name__ == "__main__":
    main()
