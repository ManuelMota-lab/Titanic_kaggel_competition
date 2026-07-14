
# Titanic Survival Prediction

[www.kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)

**Objective:** Predict which passengers survived the Titanic sinking (Kaggle binary classification competition) using the passenger manifest. The focus is on interpretable feature engineering and understanding why a model generalizes, rather than chasing a leaderboard score.

## What Was Done

- **Exploratory Data Analysis:** Validated key hypotheses—sex dominates, `Sex × Pclass` interaction is critical, cabin is largely redundant with class, and family size has a non‑linear effect.
- **Feature Engineering:**
  - Extracted `Title` from names (grouped into Mr/Mrs/Miss/Master/Rare).
  - Imputed `Age` using `Title × Pclass` medians.
  - Log‑transformed `Fare`, binned `FamilySize` (Alone/Small/Large), and flagged missingness/zero fare.
- **Modeling Pipeline:**
  - Compared Logistic Regression, Random Forest, and Gradient Boosting via cross‑validation.
  - Tuned hyperparameters with `GridSearchCV`.
  - Identified overfitting (CV vs. Kaggle gap).
  - Pruned redundant/noisy features (`HasCabin`, `Deck`, raw `Fare`, `IsAlone`) based on coefficient analysis, reducing from 32 to 21 features.

## Results

- **Final Model:** Logistic Regression (`C=0.01`)
- **Cross‑Validation Accuracy:** `0.836`
- **Kaggle Score:** `0.775`

The narrow train/val gap from the learning curve confirms stable generalization. The score sits realistically near the dataset's practical ceiling (~0.78), where outcomes are limited by unrecorded factors (e.g., physical location, individual chance).

## Files

- `titanic.ipynb` — Full analysis, modeling, and evaluation notebook.
