# MatchUp.AI: World Cup Match Predictor

This model predicts football match outcomes using Poisson regression. Data reflects historical match results, and each country's Elo ratings for current form. Trained in Python, runs entirely in the browser.

## How It Works

### Poisson Goal Model
Goals in a match are treated as random events following a Poisson distribution. The expected number of goals for a team depends on three factors: their attack strength, the opponent's defense strength, and a boost for playing at home. These values are estimated through regression on historical match results, so every team ends up with its own fitted attack and defense rating, which then influences the models' expected goals calculations.

### Elo Adjustment
Historical goal data alone doesn't reflect a team's current form. Elo ratings are incorporated to adjust each team's attack and defense strength, so the model weighs recent results more heavily than long-run history.

### Client-Side Inference
The model is trained and fitted in Python first. The resulting coefficients (attack, defense, home advantage) are then exported and reimplemented in JavaScript, allowing the app to generate full scoreline probability distributions instantly in the browser, with no backend or API calls required so I can host it statically on github pages.

## Output

For any matchup, the model returns a full probability distribution over scorelines (2-1, 1-1, 0-0, and so on). From this, win, draw, and loss probabilities, along with expected goals, can be derived.

## Data Notes

Before training, the historical match and Elo data was cleaned and validated. This included catching a mislabeled Elo entry where two national teams' ratings had been swapped, which was quietly skewing early predictions.

## Tech

- **Modeling:** Python (Poisson regression)
- **Frontend:** HTML, CSS, vanilla JS (client-side inference)
- **Styling:** Custom glassmorphism UI with animations

### Sources
- https://eloratings.net/
- https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
