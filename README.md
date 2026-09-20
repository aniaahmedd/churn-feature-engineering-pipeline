# Feature Engineering Project (5 days)

An end-to-end, leakage-safe feature engineering workflow for a **customer-churn** problem
(`churn_30d`, ~10.5% positives). Everything runs with one command and every day produces the output
named in the plan.

```bash
pip install -r requirements.txt
python run_all.py              # generates data, runs Day 1-5, then the tests   (a few minutes)
python run_all.py --day 3      # or a single day (run earlier days first at least once)
pytest -q                      # leakage / reproducibility tests only
```

> **Data:** the brief did not name a dataset, so `src/generate_data.py` creates a synthetic subscription
> dataset (customers table + 187k-row event log, dates, missing values, a noise column, a high-cardinality
> column, class imbalance). To use your own data, change only `src/data.py`
> (`load_raw` / `load_model_frame`) and the raw column lists at the top of `src/features.py`.

## Plan -> deliverable map

| Day | Task (priority) | What was built | Output |
|---|---|---|---|
| 01 | Feature Engineering | 46 numeric / date-time / interaction / categorical / aggregate features in 6 groups, feature catalog, redundancy + signal report | `src/features.py`, `src/day01_feature_engineering.py`, `outputs/day01_*` |
| 02 | Data Transformation Pipeline (High) | One `imblearn` Pipeline: FeatureEngineer -> impute/scale/one-hot -> correlation filter -> (resampler) -> model; 10 automated leakage checks | `src/pipeline.py`, `src/validation.py`, `outputs/day02_*` |
| 03 | Feature Comparison (Medium) | Ablation: baseline vs each group added, each group dropped, and each group added on top of the strongest one (paired folds, CIs) | `outputs/day03_ablation_results.csv`, `day03_ablation.png` |
| 04 | Imbalance & Robustness (High) | Imbalance check; none / class-weight / over / under / SMOTE x 2 models; 3 seeds x 5 folds; per-segment performance | `outputs/day04_*` |
| 05 | Feature Review & Documentation (High) | Permutation-importance review over 10 folds, useful/unstable/drop verdicts, final feature set, hold-out test, saved pipeline | `docs/FEATURE_DOCUMENTATION.md`, `docs/RESULTS_SUMMARY.md`, `outputs/final_pipeline.joblib` |

## How leakage is prevented ("learned only from training data")
* Every statistic that is learned across rows (median imputation, scaler mean/std, one-hot categories,
  rare-country list, promo-code frequencies, age-quintile edges, price-pressure threshold, correlation filter,
  resamplers) is a *fitted step inside one Pipeline*. Cross-validation clones the pipeline per fold.
* The target `y` is never used to build features (no target encoding).
* Event aggregates use only events dated before the snapshot and are per-customer, so they cannot leak across rows.
* 20% hold-out set is untouched until Day 05. `src/validation.py` proves the properties above
  (e.g. shuffling `y` changes nothing; fitting on train+test *would* change the statistics, so the check is meaningful).

## What the experiments found (synthetic data, logistic regression)
* **Engineered vs baseline:** PR-AUC 0.16 -> 0.44 in CV; on the hold-out 0.14 -> 0.37 (95% bootstrap CI of the
  difference excludes zero). Baseline ROC-AUC is ~0.57-0.60, engineered ~0.74-0.80.
* **Where the gain comes from:** the *aggregate* (event-log) group carries almost all of it (+0.28 PR-AUC).
  Interactions (+0.26) and categorical encodings (+0.15) look strong alone but add at most ~+0.006 once aggregates
  are present, because they are built from the same signals. Date/time features add nothing.
* **Imbalance (8.5 : 1):** class weights / resampling barely change PR-AUC (differences smaller than fold noise) but raise
  recall at the default 0.5 threshold from ~0.23 to ~0.68. Because PR-AUC is threshold-free, the final pipeline keeps the
  plain model and picks its operating threshold (0.24) from out-of-fold predictions.
* **Robustness:** seed-to-seed PR-AUC spread is 0.003-0.011; engineered features beat baseline in every segment.
  Segments with fewer than 30 positives (e.g. CA, DE, NZ, SE, device = missing) are flagged low-support.
* **Feature review:** 15 useful, 11 unstable/weak, 20 no-signal features (including the planted noise column
  `legacy_score`). Importance rankings agree only moderately between folds (Spearman 0.52) because many features
  are correlated. Pruning to the 15 useful features helped in CV, but on the hold-out it is equal within noise
  (0.368 vs 0.374), so treat pruning as a simplification, not an accuracy gain.

## Layout
```
run_all.py                 one-command runner
src/config.py              seeds, paths, snapshot date
src/generate_data.py       synthetic dataset
src/data.py                loading + train/hold-out split  (edit this for your own data)
src/features.py            DAY 1  feature engineering module + feature catalog
src/pipeline.py            DAY 2  transformation pipeline
src/validation.py          leakage / reproducibility checks
src/evaluation.py          metrics + repeated stratified CV (paired folds)
src/day0X_*.py             the five daily scripts
tests/test_pipeline.py     pytest suite (21 tests)
outputs/                   CSVs, plots, fitted pipeline, config JSON
docs/                      FEATURE_DOCUMENTATION.md, RESULTS_SUMMARY.md
```
