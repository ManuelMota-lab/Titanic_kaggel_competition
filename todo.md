## Phase 0 — Fix what's broken

1. [ ] **Cell 13.** Replace `pd.get_dummies(test_data[test_data.columns])` with `pd.get_dummies(test_data[features])`. Then delete `Name` from `features`. Your "only one survived" note in cell 14 is a misdiagnosis of this bug — rewrite that markdown once you see the real output.
2. [ ] **Cell 5.** `extract_title` takes the first token ending in `.`, which misfires on middle initials. Use `re.search(r',\s*([^.]+)\.', name).group(1).strip()`.
3. [ ] **Cell 19.** Add the missing `X_test.reindex(columns=X.columns, fill_value=0)`. It works today by luck.
4. [ ] **Collapse rare titles.** Map anything outside `{Mr, Mrs, Miss, Master}` to `Rare`. Test contains `Dona`, absent from train, currently silently zeroed by `reindex`.
5. [ ] **Delete `submission1/2/3.csv` reads.** Printing the predicted survival rate back from a CSV you just wrote is a no-op. Compute it from `predictions` if you want it at all.

## Phase 1 — Validation (the part that actually matters)

6. [ ] Set up `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` once, reuse everywhere.
7. [ ] Wrap all preprocessing in a `sklearn.pipeline.Pipeline` with `ColumnTransformer`. Right now your title medians and imputation values are computed on the full train set *before* CV splits — that's leakage into every fold. This is the most substantive methodological error in the notebook and the one most worth writing about.
8. [ ] Score every model with `cross_val_score` on that pipeline. Report **mean ± std**, not just mean. The std is what tells you whether a 0.815 vs 0.821 difference means anything.
9. [ ] Hold out a stratified 20% test split that you touch **once**, at the very end. Not for tuning.
1. [ ] Add a naive baseline: `Sex == female → survived`. It scores ~0.78. Every subsequent model must beat it, and you should say so explicitly.

## Phase 2 — Features worth building

1. [ ] `FamilySize = SibSp + Parch + 1` and `IsAlone`. Survival is non-monotonic in family size — small families fared best, large ones badly.
1. [ ] **Ticket group size.** Count passengers sharing a ticket number *across train+test combined* (this is legitimate — ticket IDs aren't the target). Groups survived or died together. This is where the real signal lives.
1. [ ] `FarePerPerson = Fare / TicketGroupSize`. Raw `Fare` is per-ticket, not per-passenger — a family of six paying £60 is not wealthy.
1. [ ] `Deck` from the first letter of `Cabin`, with `Unknown` as its own level. Keep `HasCabin` too; test whether `Deck` adds anything over it.
1. [ ] Age × Pclass interaction, or just let the tree find it. Test both, report the difference.
1. [ ] **Kill the Spearman feature selection (cell 16).** It's the wrong tool for a tree — it can't see interactions, and `Sex_female` / `Sex_male` appearing as separate rows with identical correlation is a tell that you're ranking redundant columns. Replace with permutation importance on held-out folds. Keep the old analysis in the notebook and write a paragraph on *why* you abandoned it. That paragraph is worth more than the model.

## Phase 3 — Modeling

1. [ ] Compare on identical CV folds: Logistic Regression (scaled), RandomForest, GradientBoosting or XGBoost. Three models, one table, mean ± std.
1. [ ] `GridSearchCV` or `RandomizedSearchCV` over `max_depth`, `min_samples_leaf`, `max_features` — **nested inside CV**, or your reported score is optimistic.
1. [ ] Plot a validation curve for `max_depth`. Show the overfitting elbow. Your current `max_depth=5` is an unjustified guess; make it a justified one.
2. [ ] Skip the ensemble/stacking. It'll buy you 0.003 and reads as cargo-culting.

## Phase 4 — Error analysis (the section nobody writes, and the reason to hire you)

2. [ ] Confusion matrix, precision, recall, ROC-AUC — not just accuracy. Accuracy on a 38/62 split is a weak summary.
2. [ ] Pull the ~60 misclassified passengers. Look at them individually. Patterns: 3rd-class women who died, 1st-class men who survived. Name three.
2. [ ] Calibration plot. Is `predict_proba` meaningful, or is your forest overconfident?
2. [ ] Write out where the model would fail if deployed. It has no `Cabin` for 77% of passengers; it can't see who was near a lifeboat; it's fit on 891 rows and every "improvement" past ~0.82 is noise.

## Phase 5 — Making it a portfolio artifact

2. [ ] Get it out of a single notebook. `src/features.py`, `src/model.py`, thin notebook that imports them. A 20-cell linear notebook says "I did a tutorial." A module + notebook says "I ship code."
2. [ ] `README.md`: the problem, your approach, the results table, **what you'd do differently**. Two hundred words. Recruiters read this and nothing else.
2. [ ] `requirements.txt`, pinned. Set `random_state` everywhere. Someone should be able to clone and reproduce your exact numbers.
2. [ ] Three or four charts, captioned, that each make one point. Not a wall of seaborn.
2. [ ] Write the limitations section honestly. "I got 0.82, which is within noise of the 0.79 sex-only baseline on a 418-row test set" is a *stronger* signal than claiming 0.85.
3. [ ] Then stop and build something with a dataset that isn't Titanic. One rigorous Titanic + one original project beats five tutorials.
