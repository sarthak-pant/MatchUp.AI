"""
Elo based Poisson regression model

Trains two independent GLMs (one for home gols, one for away goals) on historical international results,
using elo rating differences and a home-advantage flag as predictors. Coefficients get exported to
docs/model.json + docs/model.js for the frontend to use at runtime.

Goal counts in football are well-approximated by a Poisson distribution,
and expected goals (lambda-λ) scale with team strength (Elo) and whether
the match is at a neutral venue or not. Home and away goals are treated independently
because this allows the model to account for home advantage. 

elo_diff = (elo_home - elo_away) / 100
home_adv = 1 if on home soil, otherwise 0

"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# statsmodels' IRLS fitter can be noisy about convergence, which isn't useful here
warnings.filterwarnings("ignore")

# loading raw data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# storing international game results in results
results = pd.read_csv(DATA_DIR/"results.csv")

# storing elo ratings in elos
elos = pd.read_csv(DATA_DIR/"2026elos.csv")

# pairing elos to countries
# e.g. elo_lookup["Argentina"]--> 2148
elo_lookup = dict(zip(elos["country"], elos["elo"]))

# Adjusting elo ratings using the teams' form in this world cup's group-stage

# form_score = (z(pts_per_match) + z(gd_per_match)) / 2
# adjusted elo = elo + 30*form_score
# using 30 because it is a reasonable multiple that accounts for discrepancies between current elos and current form

points_table = pd.read_csv(DATA_DIR / "2026PointsTable.csv")

# average points per match
points_table["ppm"] = points_table["Pts"] / points_table["MP"]

# average goal difference per match
points_table["gdpm"] = points_table["GD"] / points_table["MP"]

# lambda (mean) of points per game
mean_ppm = points_table["ppm"].mean()
# standard deviation between the points per game
std_ppm = points_table["ppm"].std()
# lambda (mean) of goal difference per game
mean_gdpm = points_table["gdpm"].mean()
# standard deviation between the goal difference per game
std_gdpm = points_table["gdpm"].std()

# Calculating the Z-scores
points_table["ppm_z"] = (points_table["ppm"] - mean_ppm) / std_ppm
points_table["gdpm_z"] = (points_table["gdpm"] - mean_gdpm) / std_gdpm

# Equal weight to points and goal difference
points_table["form_score"] = (points_table["ppm_z"] + points_table["gdpm_z"]) / 2

# adjusting the elos

form_adjustments = dict(zip(points_table["Team"], points_table["form_score"] * 30))

adjustment_vals = list(form_adjustments.values())

for team in sorted(elo_lookup):
    if team in form_adjustments:
        adj = form_adjustments[team]
        old = elo_lookup[team]
        elo_lookup[team] = round(old + adj)

# Filtering data to matches involving only World cup 2026 teams from results.csv
world_cup_teams = set(elo_lookup.keys())

mask = (
    results["home_team"].isin(world_cup_teams)
    | results["away_team"].isin(world_cup_teams)
)

wc_matches = results[mask].copy()

# mergign elo ratings and build features

wc_matches["elo_home"] = wc_matches["home_team"].map(elo_lookup)
wc_matches["elo_away"] = wc_matches["away_team"].map(elo_lookup)

# Drops matches where the opponent isn't a 2026 WC team as there would be no elo to map to
wc_matches = wc_matches.dropna(
    subset=["elo_home", "elo_away", "home_score", "away_score"]
)

# elo_diff > 0 means the home team is the stronger side
wc_matches["elo_diff"] = (wc_matches["elo_home"]-wc_matches["elo_away"]) / 100

# 1 = played on the home team's soil, 0 = neutral venue
wc_matches["home_adv"] = 1 - wc_matches["neutral"].astype(int)

# Training home goals Poisson GLM

X_home = wc_matches[["elo_diff", "home_adv"]]
X_home = sm.add_constant(X_home) # adds the intercept term
y_home = wc_matches["home_score"]

# Log link keeps λ_H = exp(linear predictor) always +ve
home_glm = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()

# Training away goals Poisson GLM

# Flip the sign so a positive coefficient reads naturally: stronger away team = more away goals
wc_matches["elo_diff_away"] = -wc_matches["elo_diff"]
 
X_away = wc_matches[["elo_diff_away"]]
X_away = sm.add_constant(X_away)
y_away = wc_matches["away_score"]
 
# No home_adv term here — there's no such thing as an "away advantage".
away_glm = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()
print(away_glm.summary())

model = {
    "version": 2,
    "model-type": "elo_poisson",
    "description": (
        "Independent Poisson GLMs based on Elo rating differenced, with "
        "Elos adjusted by 2026 group-stage form (K=30)"
    ),
    # Used by frontend to populate team dropdowns
    "teams": sorted(world_cup_teams),
    # Used by the frontend to compute elo_diff for the matchup the user picks
    "elos": elo_lookup,
    "home_model": {
        "intercept": float(home_glm.params["const"]),
        "elo_diff": float(home_glm.params["elo_diff"]),
        "home_adv": float(home_glm.params["home_adv"]),
    },
    "away_model": {
        "intercept": float(away_glm.params["const"]),
        "elo_diff_away": float(away_glm.params["elo_diff_away"]),
    },
    # Not used by the frontend
    "stats": {
        "n_matches": len(wc_matches),
        "home_deviance": float(home_glm.deviance),
        "away_deviance": float(away_glm.deviance),
    },
}

# model.json — plain JSON, useful for reference or a server-based setup
DOCS_DIR.mkdir(parents=True, exist_ok=True)
json_path = DOCS_DIR / "model.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(model, f, indent=2, ensure_ascii=False)
print(f"Exported {json_path}")
 
# model.js — same data as a JS global, loaded via <script> tag so the page
# still works on file:// (fetch() would hit CORS issues there)
js_path = DOCS_DIR / "model.js"
with open(js_path, "w", encoding="utf-8") as f:
    f.write("// Model data \u2014 auto-generated by python/train_v1.py\n")
    f.write("// This file exists so the page works on file:// protocol.\n")
    f.write("const MODEL_DATA =\n")
    json.dump(model, f, indent=2, ensure_ascii=False)
    f.write(";\n")
print(f"Exported {js_path}")

#SAMPLE PREDICTIONS!

def predict_match(home_team, away_team, neutral=False):
    """
    Return (lambda_home, lambda_away) expected goals for a matchup.
    Raises KeyError if either team isn't in elo_lookup.
    """
    elo_h = elo_lookup[home_team]
    elo_a = elo_lookup[away_team]
    elo_diff = (elo_h - elo_a) / 100.0
    home_adv = 0 if neutral else 1
 
    lambda_home = np.exp(
        home_glm.params["const"]
        + home_glm.params["elo_diff"] * elo_diff
        + home_glm.params["home_adv"] * home_adv
    )
    lambda_away = np.exp(
        away_glm.params["const"]
        + away_glm.params["elo_diff_away"] * (-elo_diff)
    )
    return lambda_home, lambda_away
 
 
print("\n--- Sample predictions ---")
print(f"  {'Home':20s} vs {'Away':20s}  Neutral  lam_H  lam_A")
print(f"  {'-'*20}   {'-'*20}  ------- ------ ------")
for home, away, neutral in [
    ("Argentina", "Brazil", False),
    ("Argentina", "Brazil", True),
    ("England", "Brazil", False),
    ("Curaçao", "Argentina", True),
]:
    lh, la = predict_match(home, away, neutral)
    print(f"  {home:20s} vs {away:20s}  {neutral!s:>7}  {lh:.3f} {la:.3f}")