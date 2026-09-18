# ScoutIQ

Football analytics for Europe's top five leagues: player valuations and match
outcome predictions, with the model's own limitations measured rather than
glossed over.

- **Transfer value model** — XGBoost per position group, 15,374 player-seasons
  across the Premier League, La Liga, Serie A, Bundesliga and Ligue 1.
- **Match outcome model** — Elo with per-league calibration, plus Dixon-Coles
  for scoreline distributions. 38,983 matches, 2016 to present.

## What the numbers actually say

Test seasons 2024–25, 3,504 top-tier matches, scored once after every choice
was locked in on validation.

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Bookmaker closing odds | **0.9611** | 0.5713 | 54.4% |
| Elo + per-league calibration | 0.9932 | 0.5924 | 52.0% |
| Previous configuration | 0.9989 | 0.5962 | 51.5% |

The market is ahead by 0.032 log loss and that gap is statistically real. Four
separate attempts to close it did not:

- **More features.** A 2-parameter Elo, a 4-feature XGBoost and a 46-feature
  stacked ensemble are statistically indistinguishable from each other.
- **Dixon-Coles for outcomes.** Materially worse (1.1369). Kept only for the
  scoreline distribution, which Elo cannot produce.
- **Blending with the market.** Both linear and logarithmic pooling put weight
  **0.00** on this model. The market already contains what it knows.
- **Squad availability.** Lineup and injury data across 723k lineup rows —
  indistinguishable from baseline (−0.0005, CI [−0.0037, +0.0028]).

What *did* help: second-tier data so promoted teams arrive with real ratings,
regressing toward division rather than global means, per-league calibration,
and excluding the empty-stadium seasons from that calibration. Combined:
0.9989 → 0.9932, significant at 95%.

`/api/match/accuracy` reports live performance on fixtures predicted in
advance, which is the only number that checks the backtest against reality.

## Running locally

```bash
pip install -r requirements-dev.txt
cp .env.example .env          # add FOOTBALL_DATA_API_KEY

python src/fetch_fdcouk.py            # match results, all divisions
python src/build_match_features.py    # feature matrix
python src/build_match_predictor.py   # fit and save the match model
python notebooks/04_local_pipeline.py # train the value model

python api.py                         # API on :8000
cd frontend && npm install && npm run dev
```

`src/refresh_match_model.py` brings the match model up to date afterwards; it
re-downloads only the season in progress.

## Deploying

**API — Render.** `render.yaml` is a blueprint; point Render at the repo and it
picks it up. Set `FOOTBALL_DATA_API_KEY` and `ALLOWED_ORIGINS`. Leave every
`DATABRICKS_*` variable unset so the API serves the committed Gold CSVs.

Free tier sleeps after 15 minutes idle and takes ~50s to wake. The frontend
detects this and explains it rather than showing an error.

**Frontend — Vercel.** Root directory `frontend`. Set `VITE_API_URL` to the
Render URL — it is inlined at build time, so it must be set before the build.

**Scheduled refresh — GitHub Actions.** `.github/workflows/refresh.yml` runs
daily, refreshes data and models, and commits the result back. Render redeploys
on push. This lives in CI rather than on the host because Render's filesystem is
ephemeral and its free tier sleeps — anything written at runtime is lost.

Add `FOOTBALL_DATA_API_KEY` as a repository secret.

## Data sources

| Source | Used for | Terms |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | Results, shots, odds | Free, crawling permitted |
| [football-data.org](https://www.football-data.org/) | Fixtures, live form | Free tier, API key |
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Valuations, lineups | CC0 |
| [salimt/football-datasets](https://github.com/salimt/football-datasets) | Injury histories | Open |

FBref, FotMob and Sofascore all prohibit automated access in their terms and
are deliberately not used.

## Layout

```
api.py                     FastAPI app
scheduler.py               local scheduler (CI uses the Actions workflow)
src/
  elo.py                   ratings, division-aware season regression
  elo_calibration.py       per-league probability mapping with shrinkage
  dixon_coles.py           scoreline model, analytic gradient
  match_predictor.py       the service the API calls
  optimise_accuracy.py     the experiment that chose the configuration
  track_predictions.py     logs forecasts, scores them once results land
frontend/                  React + Vite
notebooks/                 value model pipeline
```

## A note on the numbers

Every figure here came from a held-out test set scored once, with differences
checked by paired bootstrap rather than read off the fourth decimal place.
Where something did not work, it is written down as not working. That is more
useful than a README claiming to beat the bookmaker.
