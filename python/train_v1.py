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




