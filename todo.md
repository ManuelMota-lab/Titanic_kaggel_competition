
# Titanic ML Project — TODO

## 0. Setup

- [X] Create GitHub repo `titanic-survival-prediction` (public)
- [X] `git init`, add `.gitignore` (Python template), `LICENSE` (MIT)
- [X] Create venv / conda env; `requirements.txt` with pinned versions
- [X] Folder structure:
  ```
  data/raw  data/processed  notebooks/  src/  models/  reports/figures/
  ```
- [X] Download `train.csv` / `test.csv` from Kaggle → `data/raw/`
- [X] Add `data/` to `.gitignore`, but commit a `data/README.md` explaining how to fetch it

## 1. EDA (`notebooks/01_eda.ipynb`)

- [X] Shape, dtypes, `.describe()`, missing-value counts
- [X] Target balance (survival rate ≈ 38%)
- [X] Univariate: Age, Fare distributions (log-transform Fare?)
- [X] Bivariate vs. target: Sex, Pclass, Embarked, SibSp, Parch, Age bins
- [ ] Correlation heatmap (numeric only)
- [ ] Write down 5 concrete hypotheses you'll test with features
- [ ] Save every figure to `reports/figures/`

## 2. Preprocessing & Feature Engineering (`src/features.py`)

- [X] Missing values: Age (median by Title/Pclass, not global), Embarked (mode), Fare (median), Cabin (mostly missing → derive `HasCabin` / `Deck` letter)
- [X] Extract `Title` from Name (Mr, Mrs, Miss, Master, rare→"Other")
- [X] `FamilySize = SibSp + Parch + 1`, `IsAlone`
- [X] `Deck` from Cabin first letter
- [X] Binning: `AgeBin`, `FareBin` (optional, test both)
- [ ] Encode: one-hot for low-cardinality, ordinal where ordered
- [ ] Scale numerics (needed for LR/SVM/KNN, not trees)
- [ ] **Wrap all of this in an sklearn `Pipeline` + `ColumnTransformer`** — this is what interviewers look for
- [ ] Drop leakage risk: never fit imputers/scalers on the full dataset before splitting

## 3. Validation Strategy

- [ ] `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- [ ] Decide metric: accuracy (Kaggle's metric) + report ROC-AUC, precision/recall, F1
- [ ] Build a `evaluate_model(pipeline, X, y)` helper that returns mean ± std CV score
- [ ] **Baseline first**: predict "all female survive" → record the number. Beat it or explain why not.

## 4. Modeling (`notebooks/03_models.ipynb`, `src/train.py`)

- [ ] Logistic Regression (interpretable baseline)
- [ ] Random Forest
- [ ] Gradient Boosting: XGBoost / LightGBM / CatBoost (pick one, do it well)
- [ ] SVM (RBF) — optional
- [ ] Compare all in a single results table (model | CV mean | CV std | fit time)
- [ ] Hyperparameter tuning: `RandomizedSearchCV` or Optuna on the top 2 models
- [ ] Soft-voting / stacking ensemble of the best 2–3
- [ ] Fix `random_state` everywhere

## 5. Interpretation (this is what separates you)

- [ ] Confusion matrix + ROC curve + precision-recall curve
- [ ] Permutation importance (not just `.feature_importances_`)
- [ ] SHAP: summary plot + 2 force plots on individual passengers
- [ ] Learning curve → are you over/underfitting? Would more data help?
- [ ] Error analysis: pull 10 misclassified passengers, look for a pattern, write a paragraph
- [ ] Calibration curve (optional but impressive)

## 6. Submission

- [ ] Retrain best pipeline on full train set
- [ ] Predict on `test.csv`, write `submission.csv` (`PassengerId,Survived`)
- [ ] Submit to Kaggle, record leaderboard score in README
- [ ] Sanity-check: your CV score should be within ~2-3% of LB. If not, investigate (likely overfitting to CV or a leak)
- [ ] Target: ~0.78–0.80. **Do not chase 0.85+** — those scores come from leaking the public test labels, and recruiters know it.

## 7. Code Quality

- [ ] Move logic out of notebooks into `src/` modules; notebooks only orchestrate + visualize
- [ ] Type hints + docstrings on public functions
- [ ] `black` + `ruff` (or `flake8`)
- [ ] 3–5 `pytest` tests (e.g. feature function output shape, no NaNs after transform)
- [ ] `Makefile` or `run.sh`: `make train`, `make predict`
- [ ] Config in `config.yaml`, not hardcoded constants
- [ ] Optional but strong: GitHub Actions running lint + tests on push

## 8. README (the most-read file in your repo)

- [ ] One-line project description + LB score badge at the top
- [ ] Problem statement
- [ ] Repo structure tree
- [ ] Setup instructions (`pip install -r requirements.txt`, how to get data)
- [ ] Results table (models × CV score × LB score)
- [ ] 2–3 embedded key figures (SHAP summary, confusion matrix)
- [ ] "Key findings" — 3 bullets in plain English (e.g. "Title extracted from Name was the 3rd most important feature")
- [ ] "What I'd do next" section
- [ ] Keep it under 2 screens of scrolling

## 9. Portfolio Polish

- [ ] Clear commit history (not one giant "final commit")
- [ ] Pin the repo on your GitHub profile
- [ ] Optional: write a short blog post / LinkedIn post about one thing you learned
- [ ] Optional: deploy a Streamlit/Gradio demo where someone enters passenger details and gets a prediction
