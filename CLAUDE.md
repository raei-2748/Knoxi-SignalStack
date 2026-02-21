# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SignalStack** is a data science project for the **Wharton High School Data Science Competition 2026**. The goal is to analyze hockey game data (`data/whl_2025.csv`) and produce team/player performance metrics to predict competition matchups (`data/WHSDSC_Rnd1_matchups.xlsx`).

**Team**: Ray, Harry, Derek, Eric, Darren

### ⚠️ Completion Status

The notebook currently covers **~40% of required competition deliverables**. Steps 1–4 build analytical inputs (line disparity, finishing, goalie quality). The following required outputs **do not yet exist** and must be built:

| Deliverable | Status |
|---|---|
| Phase 1a: Power rankings (32 teams, 1–32) | ❌ Not built |
| Phase 1a: Win probabilities (16 matchups) | ❌ Not built |
| Phase 1b: Top 10 line disparity teams | ✅ Built (Step 2 / Step 3) |
| Phase 1c: Visualization PNG | ❌ Not built |
| Phase 1d: Methodology writeup | ❌ Not built |

---

## Running the Notebook

The primary work lives in `Test14f1.ipynb`. It was originally built for **Google Colab** — all file paths are hardcoded to `/content/`. When running locally, either:

1. Mount or symlink the data directory to `/content/`, or
2. Update `CSV_PATH` in CELL 1 to point to `data/whl_2025.csv` and update all `/content/` output paths accordingly.

### Execution Order and Memory Dependencies

Run cells **sequentially top-to-bottom**. Later steps depend on specific variables being in memory **and** on CSV files saved by earlier steps:

```
Step 1  →  produces: long_df (memory), game_level (memory)
                     /content/game_level_step1.csv
                     /content/long_table_step1.csv

Step 2  →  RELOADS: /content/long_table_step1.csv (does NOT use long_df from memory)
           produces: disparity_table_sorted (memory)
                     /content/line_disparity_section2.csv

Step 3  →  requires: long_df (memory), disparity_table_sorted (memory)
           produces: compare_sorted (memory), adj_wide (memory)
                     /content/line_disparity_section3_adjusted.csv

Step 4  →  requires: long_df (memory)
           produces: finishing_table (memory), goalie_table (memory)
                     /content/finishing_table_step4.csv
                     /content/goalie_table_step4.csv
```

> **Important**: Step 2 deliberately re-reads from CSV, so you can re-run it standalone. Steps 3 and 4 require `long_df` to still be in memory from Step 1, and Step 3 also requires `disparity_table_sorted` from Step 2. If the kernel has been restarted, run all steps in order.

---

## Data Schema

`data/whl_2025.csv` — 25,827 rows × 26 columns. Each row is one **shift/line-matchup record** within a game.

| Key Column | Description |
|---|---|
| `game_id`, `home_team`, `away_team` | Game identifiers (each game has ~18 rows) |
| `home_off_line` / `away_off_line` | Offensive line label: `first_off`, `second_off`, `PP_up`, `PP_kill_dwn` |
| `home_def_pairing` / `away_def_pairing` | Defensive pairing label: `first_def`, `second_def`, `PP_up`, `PP_kill_dwn` |
| `home_goalie` / `away_goalie` | Goalie player ID (e.g. `player_id_142`) — one unique goalie per team |
| `toi` | Time on ice in **seconds** (not minutes) |
| `home_xg` / `away_xg` | Expected goals for the shift |
| `home_goals` / `away_goals` | Actual goals scored on the shift |
| `home_shots` / `away_shots` | Shots on goal (optional — detected automatically) |
| `home_penalty_minutes` / `away_penalty_minutes` | Penalty minutes (optional — detected automatically) |

### Dataset Facts Confirmed by Running Outputs
- 1,312 unique games (32 teams × 82 games / 2, accounting for home/away)
- Each team has **exactly one goalie** — goalie multipliers have near-zero variance (std ≈ 0.015, range 0.97–1.03), confirming goalie quality adds negligible predictive signal
- 32 unique teams (country names, lowercase)

---

## Notebook Architecture (4 Steps)

### STEP 1 — Data Loading & Reshaping (6 cells)

**What it does:**
- Cell 1: Loads raw CSV, prints shape and column names
- Cell 2: Auto-detects column names via case/punctuation-insensitive fuzzy matching into a `COLS` dict. **Edit `COLS` values directly in this cell if auto-detection fails.**
- Cell 3: Validates that each `game_id` maps to exactly one home/away team pair (anti-leakage check). Raises an error if inconsistent.
- Cell 4: Aggregates shift-level rows to **game-level** (1 row per game) by summing goals, xG, shots (if present), and penalty minutes (if present). Also computes `home_win` binary flag. Saves to `game_level_step1.csv`.
- Cell 5: Reshapes to **long format** — each original row becomes 2 rows (one for the home team perspective, one for away). This mirrors xG and goals so every row is from the perspective of the attacking team. Key columns in `long_df`: `team`, `opp`, `off_line`, `def_pair_opp`, `goalie_opp`, `toi`, `xg_for`, `goals_for`, `is_home`, `game_id`. Saves to `long_table_step1.csv`.
- Cell 6: Completion print.

**Key outputs:** `game_level` (memory), `long_df` (memory), two CSV files.

---

### STEP 2 — Raw Line Disparity (10 cells)

**What it does:**
- Reloads `long_table_step1.csv` fresh (does not use in-memory `long_df`)
- Cleans rows: drops where `toi <= 0`, `toi` missing, or `xg_for` missing
- Filters to **even-strength only**: `EV_LINES = ["first_off", "second_off"]` — power play and penalty kill rows are excluded
- Computes `xg_per60 = (total_xg / total_toi_seconds) * 3600` per `(team, off_line)` group
- Applies minimum TOI filter: `MIN_TOI = 600 * 60` seconds (600 minutes) to remove statistically unstable small samples
- Pivots to wide format: `first_line_xg60` and `second_line_xg60` columns per team
- Computes:
  - `disparity_ratio = first_line_xg60 / second_line_xg60`
  - `disparity_diff = first_line_xg60 - second_line_xg60`
- Replaces `inf` with `NaN`; drops teams where either line fails the TOI filter
- Sorts descending by `disparity_ratio`

**Key outputs:** `disparity_table_sorted` (memory), `line_disparity_section2.csv`.

---

### STEP 3 — Opposition-Adjusted Line Disparity via Poisson Regression (5 cells)

**What it does:**
- Uses `long_df` and `disparity_table_sorted` from memory (both must be present)
- Filters to even-strength lines and cleans numerics (same criteria as Step 2)
- Engineers feature: `team_offline = team + "_" + off_line` (e.g. `canada_first_off`)
- Fits a **regularized Poisson regression** modeling xG rate per second, with:
  - Features: `team_offline` (OHE), `def_pair_opp` (OHE), `goalie_opp` (OHE), `is_home` (numeric passthrough)
  - Target: `y_rate = xg_for / toi` (xG per second)
  - Exposure: `sample_weight = toi`
  - Regularization: `alpha = 0.1` (L2), `max_iter = 2000`, `tol = 1e-7`
  - Baseline (dropped) OHE category: most-frequent level per group (printed at runtime)
  - Missing `def_pair_opp` / `goalie_opp` values are filled with sentinel `"__MISSING__"` before encoding
- Extracts `beta` coefficients for `team_offline` dummies → computes `adjusted_xg60 = exp(intercept + beta) * 3600` **holding opponent and goalie at baseline conditions** (average opposition)
- Pivots to `first_line_adj_xg60` and `second_line_adj_xg60` per team
- Computes `adjusted_ratio` and `adjusted_diff`
- Merges with Step 2 raw ratios for comparison; prints correlation between raw and adjusted rankings

**Key outputs:** `compare_sorted` (memory), `adj_wide` (memory), `line_disparity_section3_adjusted.csv`.

**Note on interpretation:** `adjusted_xg60` represents the expected xG/60 for that team-line combination against a league-average opponent. The raw ratio (Step 2) and adjusted ratio (Step 3) will be highly correlated because L2 regularization shrinks coefficients toward zero, so opponent adjustment has limited impact with this dataset structure.

---

### STEP 4 — Finishing Skill & Goalie Quality via Poisson Regression (5 cells)

**What it does:**
- Uses `long_df` from memory
- Maps `off_line` → `situation` using an **editable dict** `OFFLINE_TO_SITUATION`:
  - `first_off` → `"EV"`, `second_off` → `"EV"`, `PP_up` → `"PP"`, all others → `"OTHER"`
- Filters to: `xg_for > 0` (required as exposure), `goals_for` not missing, `situation` in `["EV", "PP"]` (excludes penalty kill and empty net)
- Fits a **regularized Poisson regression** modeling goals-per-xG conversion rate:
  - Features: `team` (OHE), `goalie_opp` (OHE), `situation` (OHE), `is_home` (numeric passthrough)
  - Target: `y_rate = goals_for / (xg_for + 1e-6)`
  - Exposure: `sample_weight = xg_for`
  - Regularization: `alpha = 0.2` (L2), `max_iter = 3000`, `tol = 1e-7`
- Extracts:
  - `finishing_multiplier` per team: `exp(beta_team)` — values > 1 mean the team converts xG above expectation
  - `goalie_multiplier` per goalie: `exp(beta_goalie)` — **lower is better** (a goalie with multiplier < 1 suppresses opponent conversion; goalie_table sorted ascending)
- Extreme value warnings fire if any multiplier falls outside `[0.5, 2.0]`

**Key outputs:** `finishing_table` (memory), `goalie_table` (memory), `finishing_table_step4.csv`, `goalie_table_step4.csv`.

**Critical empirical finding:** Goalie multipliers from actual run: mean ≈ 1.000, std ≈ 0.015, range 0.97–1.03. This confirms that after L2 regularization, **goalie quality has essentially no differentiating signal in this dataset**. The `goalie_multiplier` output should not be weighted heavily in any downstream power ranking model.

---

## Key Modeling Decisions

### Even-strength filter
`EV_LINES = ["first_off", "second_off"]` — power play and penalty kill units are excluded from line disparity analysis (Steps 2 and 3) because special teams performance is not comparable to 5-on-5 play. Step 4 does include `PP_up` rows for the finishing model but excludes `PP_kill_dwn` and empty net.

### Exposure handling (sklearn limitation)
`sklearn.linear_model.PoissonRegressor` does not natively support a log-offset term. Both models approximate this using the rate trick:
- **Step 3**: `y_rate = xg_for / toi`, `sample_weight = toi`
- **Step 4**: `y_rate = goals_for / xg_for`, `sample_weight = xg_for`

This is mathematically equivalent to a Poisson model with log offset under certain assumptions but is an approximation. A proper offset would require `statsmodels`.

### Baseline category selection
In all OHE transformations, the dropped (baseline) category is the **most-frequent level** in the training data. This is printed at runtime for transparency. The baseline team's `finishing_multiplier` and the baseline goalie's `goalie_multiplier` are both exactly 1.0 by construction — all other multipliers are relative to that baseline.

### `__MISSING__` sentinel
Null values in `def_pair_opp` and `goalie_opp` are replaced with the string `"__MISSING__"` before OHE encoding. These rows still contribute to model training. The `__MISSING__` goalie is excluded from printed leaderboards but retained in the saved CSV.

### Column auto-detection
`CELL 2` of Step 1 auto-maps column names via normalized string matching. If the CSV schema changes, edit the `COLS` dict values directly in that cell (set the value to the exact column name string from the CSV).

---

## Intermediate File Reference

All intermediate files are written to `/content/` in Colab:

| File | Produced By | Contents |
|---|---|---|
| `game_level_step1.csv` | Step 1, Cell 4 | 1 row/game: goals, xG, shots, PIM totals + `home_win` |
| `long_table_step1.csv` | Step 1, Cell 5 | 2 rows/record: team perspective with `xg_for`, `goals_for`, `off_line`, `def_pair_opp`, `goalie_opp`, `is_home` |
| `line_disparity_section2.csv` | Step 2, Cell 10 | Per-team: `first_line_xg60`, `second_line_xg60`, `disparity_ratio`, `disparity_diff`, sorted by `disparity_ratio` desc |
| `line_disparity_section3_adjusted.csv` | Step 3, Cell 5 | Per-team: raw and adjusted ratios/diffs side-by-side |
| `finishing_table_step4.csv` | Step 4, Cell 5 | Per-team `finishing_multiplier`, sorted desc |
| `goalie_table_step4.csv` | Step 4, Cell 5 | Per-goalie `goalie_multiplier`, sorted asc (lower = better) |

---

## What Still Needs to Be Built

The following pipeline components are required for competition submission and are **absent from the notebook**:

### Step 5 (to be built): Team Power Rankings
Using `game_level_step1.csv` and `finishing_table_step4.csv`, compute per-team:
- xG differential per game (xG for minus xG against, divided by games played)
- Win percentage
- Optional: composite score incorporating finishing multiplier

Output: 32 teams ranked 1–32 by overall strength.

### Step 6 (to be built): Win Probability Predictions
Using team strength ratings from Step 5, predict home team win probability for all 16 matchups in `WHSDSC_Rnd1_matchups.xlsx` using:

$$p = \sigma\!\bigl(k \cdot (\text{rating}_{\text{home}} - \text{rating}_{\text{away}}) + \text{home\_adv}\bigr)$$

where $\sigma$ is the logistic function, $k$ is a scale factor, and `home_adv` is calculated from the actual home win rate in `game_level_step1.csv` (empirical, not assumed).

All probabilities must be clipped to $[0.01, 0.99]$.

### Step 7 (to be built): Visualization
Scatter plot of `disparity_ratio` (x-axis, from Step 2 or 3) vs. team strength rating (y-axis, from Step 5), one point per team, color-coded by tier, with regression line and R² annotation. Export as PNG ≤ 5MB.