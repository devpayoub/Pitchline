# How to Run AIFootballPredictions (Windows, verified)

Every command below was actually executed in this folder and works. Run all commands
from the project root (`AIFootballPredictions-main`), because the scripts use relative paths.

A ready-to-use virtual environment already exists at `.venv/` (Python 3.12) with a pinned
dependency stack that can load the pre-trained `.pkl` models.

---

## 1. Quick start — just get predictions (2 minutes)

The repo already ships `data/processed/*.csv`, `models/*.pkl` and `data/next_matches.json`,
so you can skip acquisition/preprocessing/training entirely.

```bash
# from the project root
.venv/Scripts/python.exe scripts/make_predictions.py \
  --input_leagues_models_dir models \
  --input_data_predict_dir data/processed \
  --final_predictions_out_file final_predictions.txt \
  --next_matches data/next_matches.json
```

Output goes to `final_predictions.txt` (same format as the README table).

> Note: the flag names in the README (`--models_dir`, `--data_dir`, `--output_file`,
> `--json_competitions`) do **not** match the script. Use the four flags above.

### Windows cmd.exe vs PowerShell vs Git Bash

The commands in this guide are written for **Git Bash / PowerShell**, which accept forward
slashes. In **cmd.exe** a leading `.venv/Scripts/python.exe` is read as a single command name and
fails with `'.venv' is not recognized as an internal or external command`. In cmd use backslashes:

```bat
.venv\Scripts\python.exe scripts\make_predictions.py --help
```

Or activate the environment once per window, then just call `python`:

```bat
.venv\Scripts\activate.bat
python scripts\make_predictions.py --help
```

### Ready-made wrappers (double-click friendly)

| File | What it does | Time |
| --- | --- | --- |
| `predict.bat` | refresh fixtures (API + Europa League) and rebuild `final_predictions.txt` | ~30 s |
| `predict_one.bat` | **numbered menu**: pick one competition (or all) → `predictions_<CODE>.txt` | instant |
| `predict_one.bat E0` | same, without the menu (any letter case, e.g. `sp1`) | instant |
| `train_all.bat` | full pipeline: download, pool, preprocess, train both markets, predict | 30-40 min |

The menu looks like this:

```
==================================================
   AI Football Predictions - pick a competition
==================================================
   1) Champions League      CL
   2) Europa League         EL
   3) Premier League        E0
   4) Serie A               I1
   5) Bundesliga            D1
   6) La Liga               SP1
   7) Ligue 1               F1
   8) All competitions
   0) Exit
```

They `cd` to their own folder, so they work from any working directory, including a double-click.

---

## 2. Recreate the environment from scratch

`conda/aifootball_predictions.yaml` does **not** work as-is: it pins `tensorflow==2.10`
(Python 3.10 only, no Windows wheels for newer versions) and leaves `scikit-learn`
unpinned, which breaks unpickling of the shipped models. A plain venv with pinned
versions is simpler and verified:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install ^
  "numpy==1.26.4" "scipy==1.11.4" "pandas==2.2.3" "scikit-learn==1.5.1" ^
  "xgboost==2.1.4" "category_encoders==2.6.4" ^
  mrmr_selection scikit-optimize seaborn requests python-dotenv invoke
```

Why these pins (each one is a real failure I hit):

| Package | Pin | Reason |
| --- | --- | --- |
| `scikit-learn` | `1.5.1` | The `.pkl` files were pickled with sklearn **1.5.1**. On 1.6+ you get `AttributeError: __pyx_unpickle_CyHalfBinomialLoss`; on 1.9 `ModuleNotFoundError: sklearn._loss`. |
| `scipy` | `1.11.4` | `data_preprocessing.py` calls `sch.distance.pdist`. Newer SciPy (1.18) no longer exposes the `distance` submodule as an attribute → `module 'scipy.cluster.hierarchy' has no attribute 'distance'`. |
| `category_encoders` | `2.6.4` | `mrmr_selection` pulls the latest version, which needs sklearn ≥ 1.6 (`ImportError: cannot import name 'Tags'`). |
| `numpy` | `1.26.4` | Required by the SciPy 1.11.4 / sklearn 1.5.1 pair. |
| `tensorflow` | *not needed* | No script imports it; it is only listed in the conda yaml. |

---

## 3. Full pipeline, stage by stage

### 3.1 Data acquisition (needs internet, no key)

```bash
.venv/Scripts/python.exe scripts/data_acquisition.py \
  --leagues E0 I1 SP1 F1 D1 --seasons 2627 2526 2425 2324 \
  --raw_data_output_dir data/raw
```

Downloads from football-data.co.uk and writes `data/raw/<league>_merged.csv`.
Season codes are validated against `VALID_SEASONS` in the script, which now accepts
`2627 2526 2425 2324 2223 2122 2021` (2026-27 is the current season).

### 3.2 Preprocessing

```bash
.venv/Scripts/python.exe scripts/data_preprocessing.py \
  --raw_data_input_dir data/raw --processed_data_output_dir data/processed \
  --num_features 20 --clustering_threshold 0.5
```

Writes `data/processed/<league>_merged_preprocessed.csv` (24 columns: 14 mRMR-selected
features + identifiers + the `Over2.5` target). Verified: reproduces the shipped files.

### 3.3 Training

```bash
.venv/Scripts/python.exe scripts/train_models.py \
  --processed_data_input_dir data/processed --trained_models_output_dir models \
  --metric_choice accuracy --n_splits 10 --voting soft
```

Trains LR / KNN / SVM / RF / XGBoost / HistGB with BayesSearchCV, then a soft-voting
ensemble per league, saved as `models/<league>_voting_classifier.pkl`.
Slow with `--n_splits 10`; `--n_splits 3` gives a quick smoke test (~30 s per league).

### 3.4 Next matches (needs an API key)

1. Register at https://www.football-data.org/ and copy your token.
2. Put the key in `C:\Users\<you>\.env` (the **home** directory, not the project) containing:
   ```
   API_FOOTBALL_DATA=your_token_here
   ```
   `acquire_next_matches.py` calls `load_dotenv(os.path.expanduser("~/.env"))`, so a `.env`
   sitting in the project root is ignored. Copy it: `cp .env ~/.env`.
3. Run:
   ```bash
   .venv/Scripts/python.exe scripts/acquire_next_matches.py \
     --get_teams_names_dir data/processed --next_matches_output_file data/next_matches.json
   ```
   This overwrites `data/next_matches.json` with the upcoming fixtures. Add
   `--window_days 16` to reach the next round when a competition has no games this week
   (the Champions League is empty for 7 days right after a matchday). The free tier allows
   10 requests/minute and this step makes six, so the script now retries with a 20 s backoff
   when it gets a 429 instead of crashing.

### 3.5 Predictions

```bash
.venv/Scripts/python.exe scripts/make_predictions.py \
  --input_leagues_models_dir models --input_data_predict_dir data/processed \
  --final_predictions_out_file final_predictions.txt --next_matches data/next_matches.json
```

### 3.6 Champions League (pooled European model)

football-data.co.uk has **no** Champions League file (`CL.csv`, `UCL.csv` both 404), and the
football-data.org API serves CL fixtures but no shots, corners or bookmaker odds — the exact
inputs the feature engineering needs. So the CL model is not trained on CL matches. Instead:

1. Download five extra European leagues (only needed once, and after a season change):
   ```bash
   .venv/Scripts/python.exe scripts/data_acquisition.py \
     --leagues N1 B1 P1 T1 G1 --seasons 2627 2526 2425 2324 --raw_data_output_dir data/raw
   ```
2. Pool all ten leagues into one dataset:
   ```bash
   .venv/Scripts/python.exe scripts/build_european_dataset.py
   ```
   Writes `data/raw/CL_merged.csv` (~10,450 matches).
3. Preprocess and train just that file:
   ```bash
   .venv/Scripts/python.exe scripts/data_preprocessing.py \
     --raw_data_input_dir data/raw --processed_data_output_dir data/processed \
     --num_features 20 --clustering_threshold 0.5
   .venv/Scripts/python.exe scripts/train_models.py \
     --processed_data_input_dir data/processed --trained_models_output_dir models \
     --metric_choice accuracy --n_splits 5 --voting soft
   ```
   Produces `models/CL_voting_classifier.pkl` (5-fold accuracy ≈ 0.79).
4. Run steps 3.4 and 3.5 — `CL` is already in `VALID_LEAGUES` and `COMPETITIONS`.

Caveats specific to CL:
- Only fixtures where **both** clubs exist in the ten pooled leagues can be scored. Shakhtar,
  LASK, Viking, Bodø/Glimt, Slavia, Slovan and Sabah have no data, so their matches are skipped.
- CL games are much more open than the domestic matches the model learned from: 2026-27 CL
  averaged 4.16 goals and 73.3% over 2.5, versus 2.84 and 54.4% in the pooled domestic data.
  `make_predictions.py` therefore applies a base-rate shift (`PRIOR_SHIFT["CL"] = 2.30`) to CL
  probabilities. Set it to `1.0` to see the raw model output.

### 3.7 Europa League (same model, different fixture source)

The Europa League is **not** reachable from the football-data.org free tier — competition 2143
returns `403 restricted`, and TheSportsDB blocks its free key too. Fixtures are therefore parsed
from the Wikipedia league-phase page, which stores them in `{{#invoke:Football box|main}}`
templates:

```bash
.venv/Scripts/python.exe scripts/fetch_europa_league.py --window_days 7
```

That writes an `EL` key into `data/next_matches.json` (merging, not overwriting the other
leagues), then step 3.5 produces the report as usual. `EL` reuses the pooled CL model via
`MODEL_ALIAS = {"EL": "CL"}` in `make_predictions.py`, with its own base-rate shift
(`PRIOR_SHIFT["EL"] = 1.48`, from a 63.9% over-2.5 rate across the 36 played EL matches).

Coverage is the weak point: EL draws clubs from leagues the pooled dataset does not contain
(Armenia, Croatia, Cyprus, Czechia, Hungary, Israel, Slovenia, Bulgaria). On the current matchday
only **4 of 18** fixtures have both clubs in the training data. Extending coverage means adding
more leagues, and note football-data's "extra leagues" files (Austria, Poland, Norway…) use a
completely different column layout than the main-league files, so they need normalising first.

## 4. Two markets: Over/Under 2.5 and 1X2
Both targets are always written to `data/processed/*.csv`:

| Column | Meaning |
| --- | --- |
| `Over2.5` | `1` if `FTHG + FTAG > 2`, else `0` — binary |
| `Result` | `0` = home win, `1` = draw, `2` = away win — three classes |

`--target` selects which one drives mRMR feature selection and which one the models learn:

```bash
# over/under 2.5  -> models/<league>_voting_classifier.pkl
.venv/Scripts/python.exe scripts/train_models.py \
  --processed_data_input_dir data/processed --trained_models_output_dir models \
  --metric_choice accuracy --n_splits 5 --voting soft --target over25

# 1X2             -> models/<league>_result_classifier.pkl
.venv/Scripts/python.exe scripts/train_models.py \
  --processed_data_input_dir data/processed --trained_models_output_dir models \
  --metric_choice accuracy --n_splits 5 --voting soft --target result
```

The same switch exists in preprocessing if you want the feature selection itself to be
optimised for 1X2 (features that explain results differ from features that explain goals):

```bash
.venv/Scripts/python.exe scripts/data_preprocessing.py \
  --raw_data_input_dir data/raw --processed_data_output_dir data/processed \
  --num_features 20 --clustering_threshold 0.5 --target result
```

`make_predictions.py` needs no extra flag: if `models/<league>_result_classifier.pkl` exists it
adds a second line under each match, otherwise it prints only the over/under call.

```
- ⚽ **Arsenal** 🆚 **Everton**: Over 2.5 Goals! 🔥 (91.5% chance)
   └ 1X2: Arsenal 61.2% | Draw 22.4% | Everton 16.4%
```

Things to expect with 1X2:
- Accuracy is much lower than over/under — 60-67% here versus ~80-85%. Compare against the
  always-back-the-home-side baseline (~45%), not against 100%.
- The three probabilities sum to 100%, so the model can look "confident" while being wrong.
  Treat it as a ranking signal, not a promise.
- The CL 1X2 model is the weakest: it is trained on the two most recent pooled seasons
  (3,887 matches) because the full 10k-row three-class search took over 20 minutes. Its
  extreme-looking outputs (e.g. 5% for a big away side) are model overconfidence, not a bug.

### Training speed notes

`train_models.py` now runs the search with `n_jobs=-1` and pins `n_jobs=1` inside
`XGBClassifier`, because XGBoost otherwise spawns one thread per core *inside* each of the
parallel workers and the two levels of parallelism fight each other. Without that, the
10k-row CL search alone ran past 20 minutes on one estimator.

Two 1X2-specific fixes worth knowing about:
- `HistGradientBoostingClassifier` runs with `early_stopping=False` for **every** target. With
  early stopping on, it does its own stratified split inside `HalvingGridSearchCV`'s small first
  slice and dies with "The least populated class in y has only 1 member" (1X2) or
  "test_size = 1 should be greater or equal to the number of classes" (binary with `--n_splits 3`).
- The multiclass SVM grid drops the `poly` kernel. `SVC(probability=True)` on three classes
  means one-vs-one plus internal Platt calibration, which made poly kernels impractical.

The models currently in `models/` were trained before the HGB change, so a fresh retrain will
produce slightly different numbers. Both versions are valid; just do not mix them when comparing
runs.

---

## 5. Running one league at a time

Every stage accepts `--leagues` (one or more codes). Without it, the stage runs on every
file it finds in the folder.

```bash
# 1. data for one league only
.venv/Scripts/python.exe scripts/data_acquisition.py \
  --leagues E0 --seasons 2627 2526 2425 2324 --raw_data_output_dir data/raw

# 2. preprocess just that league (~8-20 s)
.venv/Scripts/python.exe scripts/data_preprocessing.py \
  --raw_data_input_dir data/raw --processed_data_output_dir data/processed \
  --num_features 20 --clustering_threshold 0.5 --leagues E0

# 3. train just that league (~1-3 min at --n_splits 3)
.venv/Scripts/python.exe scripts/train_models.py \
  --processed_data_input_dir data/processed --trained_models_output_dir models \
  --metric_choice accuracy --n_splits 3 --voting soft --target over25 --leagues E0

# 4. predict just that league (instant)
.venv/Scripts/python.exe scripts/make_predictions.py \
  --input_leagues_models_dir models --input_data_predict_dir data/processed \
  --final_predictions_out_file predictions_E0.txt \
  --next_matches data/next_matches.json --leagues E0
```

Supported codes for prediction: `CL`, `EL`, `E0`, `I1`, `D1`, `SP1`, `F1`. An unknown code
stops the run with a message listing the valid ones.

Typical timings on this machine (12 cores):

| Stage | One league | All leagues |
| --- | --- | --- |
| acquisition | ~5 s | ~25 s |
| preprocessing | 8-20 s | ~1 min |
| training, `--n_splits 3` | 1-3 min | ~12 min (6 competitions) |
| training, `--n_splits 5` | 2-5 min | ~25 min |
| prediction | instant | instant |

Per-league notes:
- **E0, I1, D1, SP1, F1** — fully independent, train and predict on their own.
- **CL** — not a domestic league: it needs the pooled dataset from step 3.6
  (`build_european_dataset.py`), and retraining it touches only the `CL_*` files.
- **EL** — has no model of its own; it reuses `CL` via `MODEL_ALIAS`. Refresh its fixtures
  with `fetch_europa_league.py` and then predict with `--leagues EL`.

## 6. Known quirks

- **Fixed in this repo:** `acquire_next_matches.py` used to select only fixtures whose
  `matchday == season.currentMatchday`. Since that round is normally already played, every
  fixture was filtered out and `next_matches.json` came back with 0 matches per league.
  It now collects every `SCHEDULED`/`TIMED` fixture within `NEXT_MATCHES_WINDOW_DAYS`
  (default 7, top of the file).
- **Also extended:** `TEAMS_NAMES_MAPPING` gained the clubs that appear in the refreshed
  seasons but were missing (Sassuolo, Frosinone, FC Koln, Schalke 04, Lorient, Troyes,
  Elche, Leeds, Sunderland, Coventry, Hull, Hamburg, Paderborn, Elversberg, Levante,
  Malaga, Le Mans). Only Deportivo and Racing Santander are still unmatched, because they
  have no rows in `data/processed` yet.
- **Data coverage:** `data/raw` and `data/processed` currently span **Aug 2023 → 19 Oct 2026**
  (seasons 2324, 2425, 2526, 2627). Re-run stages 3.1 → 3.3 whenever you want fresher form.
  Backups of the original 2025 artefacts live in `models_backup_2025-02-28/` and
  `data/processed_backup_2024-09-30/`.
- `scripts/tasks.py` (`invoke`) is **out of sync** with `make_predictions.py`: its
  `make_predictions` task passes `--models_dir/--data_dir/--output_file/--json_competitions`,
  which the script rejects. Run the scripts directly, or fix that one task first.
- `data/next_matches.json` is read/written as **UTF-16**. If you hand-edit it in an editor,
  keep the encoding, or `json.load` fails.
- A league section can be empty when the fixture list has no teams matching that league's
  processed data.

---

## 7. Control panel (web UI)

A local web app wraps the three stages into three buttons. Nothing leaves the machine:
the server binds `127.0.0.1` only.

```bat
run_panel.bat            :: double-click friendly — starts the server and opens the browser
```

or manually:

```bash
.venv/Scripts/python.exe webapp/server.py      # -> http://127.0.0.1:8000
```

| Button | Runs | Rough time |
| --- | --- | --- |
| **Download data** | both `data_acquisition.py` calls → `build_european_dataset.py` → `data_preprocessing.py` → fixtures (7-day) → CL fixtures (16-day) → `merge_cl.py` → `fetch_europa_league.py` | ~6 min |
| **Train models** | `train_models.py --target over25` (5 folds) → `train_models.py --target result` (3 folds) → `train_additional_targets.py` | ~35 min |
| **Predict & compare** | `make_predictions.py` → `make_predictions_full.py` → `export_results.py model` → `fetch_odds_v4.py` → `export_results.py merge` → `build_report_page.py` | ~1–3 min |

Each step is a real subprocess in the project root, so the console panel shows exactly what
a terminal would. `Stop` terminates the running step; the job ends as `stopped`.

### Where the data flows

```
models/*.pkl + data/processed/*.csv
        │
        ├─ export_results.py model ──► data/results_model.json   (7 markets, no odds)
        │                                        │
        │                                        ▼
        │                            fetch_odds_v4.py  ──► data/odds.json
        │                            (1X2 · O/U 2.5 · BTTS, de-vigged per bookmaker)
        │                                        │
        └────────────────────────────────────────┴──► export_results.py merge
                                                              │
                                                              ▼
                                                      data/results.json
                                                       ├──► the control panel UI
                                                       └──► build_report_page.py
                                                              └──► data/predictions_report.html
```

### The 1X2 odds mapping gotcha

Flashscore's `HOME_DRAW_AWAY` rows identify each pick by an **event-participant id**, which is
*not* the team id from the day feed and does not match `home`/`away` ids. `fetch_odds_v4.py`
resolves them with the `dsos2` query (`findEventById` → `eventParticipants[].type.side`); the
selection with no participant id is the draw. Do not fall back to positional order — it is
correct for some fixtures and wrong for others.

Fixture matching is strict on purpose: **both** sides must reach 0.72 similarity
(`MIN_SIDE_SIM`), otherwise a shared club name binds the wrong game. Before that guard existed,
`Galatasaray vs Stuttgart` matched `Paderborn vs Stuttgart` and produced a nonsense 72% edge.
Fixtures outside the 8-day odds window (e.g. a Champions League round two weeks out) now
correctly come back as "no odds" instead of matching a random domestic game.

---

## 8. Prediction history, grading and honest accuracy

Every `Predict & compare` run now ends with `archive_predictions.py`, which writes a snapshot to
`data/history/predictions_<timestamp>.json` and keeps `data/history/index.json`. Nothing is
overwritten, so a prediction can always be graded against the score it was made for.

```
Predict & compare  ->  data/results.json  ->  archive  ->  data/history/predictions_*.json
Grade results      ->  finished scores    ->  grade    ->  data/history/graded.json
Train models       ->  ...                ->  backtest ->  data/history/backtest.json
```

### Grading rules

| Market | Pick | Correct when |
| --- | --- | --- |
| 1X2 | highest of home / draw / away | matches the final result |
| O/U 1.5 · 2.5 · 3.5 | side with probability ≥ 50% | total goals on the right side of the line |
| BTTS | yes / no | both teams scored (or not) |
| Value bets | the archived selection | that selection won |

A fixture is only graded when the result lands **on or after kick-off and within 3 days**. This is
not optional: `data/raw` dates are `dd/mm/yyyy`, and a naive `strptime("%Y-%m-%d")` throws, gets
swallowed, and silently grades today's fixture against a meeting from two seasons ago. That bug
produced 47 "graded" predictions on fixtures that had not been played yet.

### Out-of-time backtest

`backtest_models.py` sorts each league by date, trains on the first 80% and scores the last 20%.
That is the number to quote. Cross-validation inside `train_models.py` is higher because the folds
mix past and future matches.

Current numbers (2026-10-22, n = 1929):

| Market | Out-of-time | Always-pick | Lift |
| --- | --- | --- | --- |
| 1X2 | 54.4% | 41.8% | +12.6 |
| O/U 2.5 | 67.6% | 56.8% | +10.8 |
| O/U 1.5 | 78.7% | 78.4% | +0.3 |
| O/U 3.5 | 70.1% | 67.0% | +3.2 |
| BTTS | 67.1% | 57.5% | +9.6 |

O/U 1.5 has no lift: over 1.5 already happens 78% of the time, so the model is just reproducing the
base rate. Do not bet that market on the strength of a 78% figure.

## 9. Why the old models were "83% accurate" — and what changed

Three separate leaks, all fixed on 2026-10-22:

1. **Target columns as features.** `train_models.py` dropped only `Over2.5` and `Result`, so `BTTS`,
   `Over1.5` and `Over3.5` — all derived from the same scoreline — stayed in the feature matrix.
2. **Rolling features without `.shift(1)`.** `Last5HomeOver2.5Perc`, `HomeOver2.5Perc`,
   `AvgHomeGoalsScored` … all included the match they were attached to.
3. **`head(5)` instead of `tail(1)`.** The prediction row was built from a team's *oldest* five
   matches, so "current form" was actually form from the start of the dataset.

Together these explain the 80–84% cross-validation accuracy and the 92–99% probabilities. After the
fix: O/U 2.5 CV ≈ 63–70%, 1X2 ≈ 55–59%, and predicted probabilities span a believable range.

Feature selection now lives in one place (`scripts/features.py`), and every model is saved with a
`<league>_<kind>_features.json` manifest. Prediction reads the manifest, so training and prediction
cannot drift apart — the old failure mode was
`X has 16 features, but LogisticRegression is expecting 19`.

## 10. Europa League coverage

`fetch_europa_league.py` always fetches **all** fixtures; the old prediction step then dropped any
match whose clubs had no history in the pooled dataset, which is why only 4 of 18 appeared.

Now every fixture gets a row. Clubs with no history (Armenia, Croatia, Cyprus, Czechia, Hungary,
Israel, Slovenia, Bulgaria, Scotland …) fall back to competition priors from `export_results.BASELINE`
and are marked `source: "baseline"` — shown with a blue **baseline** tag in the UI. They are honest
priors, not model output, and they still get compared with the live market price.

The proper fix is to add those leagues to the pooled dataset. football-data.co.uk does not carry
most of them, and its "extra leagues" (Austria, Poland, Norway …) use a different column schema, so
it needs a normaliser, not just a download.

## 11. Live tracking

`scripts/live_predictions.py` reads the newest archive and the live Flashscore feed, then
grades every market against the **current** score. The panel exposes it at `/api/live`
(scraped on demand, cached 45 s) and the UI polls it every 45 s.

The point is that a goals market does not have to wait for the final whistle:

| Market | Settles early when | Otherwise |
| --- | --- | --- |
| Over 1.5 | total ≥ 2 | Under settles only at FT |
| Over 2.5 | total ≥ 3 | Under settles only at FT |
| Over 3.5 | total ≥ 4 | Under settles only at FT |
| BTTS yes | both teams have scored | BTTS no settles only at FT |
| BTTS no | — | at FT |
| 1X2 | — | always at FT (a 3-0 lead is not a result) |

So a 0-2 after 20 minutes already puts a green tick on "Over 1.5" while the match is running,
and a red cross on "Under 1.5". The Live tab shows a card per fixture with one chip per
market, and the same ✓ / ✗ appears inline in the 1X2, Goals, BTTS and Value bets tables.

### Team-name matching

`scripts/teams.py` is now the single matcher used by the odds fetcher, the grader and the live
tracker. It strips accents (`Plzeň` → `plzen`, not `plze`) and expands aliases. Without that,
`OFI` vs `OFI Crete`, `St. Gilloise` vs `Royale Union SG` and `Olympiakos` vs
`Olympiacos Piraeus` were silently dropped — 12 of 18 Europa League fixtures were missing from
the live board.

### Note on polling

Flashscore is a free, unauthenticated feed. The server caches a scrape for 45 s, and the page
polls on the same interval, so one open tab costs about 1.3 requests a minute. Do not lower
that much further.
