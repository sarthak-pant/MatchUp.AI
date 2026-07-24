# World Cup Match Predictor — Poisson GLM

A statistical model that predicts football match outcomes using a Poisson regression on team attack/defense strength, adjusted with Elo ratings.
Trained in Python, exported and run entirely client-side in JavaScript.

## How It Works

### 1. Poisson Goal Model
Goals scored by each team in a match are modeled as independent Poisson random variables.
The expected number of goals for a team depends on:

- **Attack strength** — how many goals the team tends to score
- **Opponent's defense strength** — how many goals the opponent tends to concede
- **Home advantage** — a fixed boost applied to the home team

These parameters are estimated via regression on historical match results, so every team ends up with a fitted attack and defense coefficient.

### 2. Elo Rating Adjustment
Raw historical goal counts don't reflect a team's *current* form.
Elo ratings are incorporated to adjust the base attack/defense strengths, so a team's predicted performance reflects recent results, not just long-run history.

### 3. Client-Side Inference
The model is trained and fitted in Python, then the resulting parameters (attack/defense coefficients, home advantage) are exported and reimplemented in JavaScript.
This lets the app compute full scoreline probability distributions instantly in the browser, with no server call or API dependency.

## Output

For any matchup, the model returns a probability distribution over scorelines (e.g. P(2-1), P(1-1), P(0-0), ...), from which win/draw/loss probabilities and expected goals can be derived.

## Data Notes

Historical match data and Elo ratings were validated and cleaned prior to training — including correcting a mislabeled Elo entry that had swapped two national teams' ratings, which was silently skewing early predictions.

## Tech

- **Modeling:** Python (Poisson regression)
- **Frontend:** HTML, CSS, JavaScript (vanilla, client-side inference)
- **Styling:** Custom glassmorphism UI with animations
