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

