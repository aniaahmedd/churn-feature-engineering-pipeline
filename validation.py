"""Automated leakage / robustness checks for the pipeline (used by Day 02, Day 05 and the tests)."""
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .config import RANDOM_STATE, SNAPSHOT_DATE
from .evaluation import cv_evaluate
from .features import (CATEGORICAL_FEATURES, FEATURE_CATALOG, FeatureEngineer, build_aggregate_features)
from .pipeline import build_pipeline

_LEARNED = ["age_edges_", "country_keep_", "promo_freq_", "price_pressure_threshold_"]


def _learned_state(fe):
    return {k: getattr(fe, k) for k in _LEARNED}


def _same(a, b):
    return all(np.array_equal(np.asarray(a[k], dtype=object), np.asarray(b[k], dtype=object))
               if not isinstance(a[k], dict) else a[k] == b[k] for k in a)


def run_checks(X_train, y_train, X_test, events=None, pipeline_kwargs=None):
    """Returns a list of {check, passed, detail}."""
    kw = pipeline_kwargs or {}
    results = []

    def add(name, passed, detail=""):
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1. learned statistics come from the training data only
    fe_train = FeatureEngineer().fit(X_train)
    state_before = _learned_state(fe_train)
    fe_train.transform(X_test)                                   # transforming test data must not change state
    add("fit_learns_only_from_training_data", _same(state_before, _learned_state(fe_train)),
        "transform(X_test) left every learned statistic unchanged")

    fe_all = FeatureEngineer().fit(pd.concat([X_train, X_test]))
    add("sensitivity_check_leaky_fit_differs",
        not _same(state_before, _learned_state(fe_all)), "fitting on train+test WOULD change the learned statistics, so the check above is meaningful")

    # 2. the target is never used while learning features
    fe_a = FeatureEngineer().fit(X_train, y_train)
    fe_b = FeatureEngineer().fit(X_train, y_train.sample(frac=1, random_state=0).reset_index(drop=True))
    add("features_do_not_use_target", _same(_learned_state(fe_a), _learned_state(fe_b)),
        "shuffling y does not change any learned value")

    # 3. scaler statistics equal the training-fold statistics
    pipe = build_pipeline(**kw).fit(X_train, y_train)
    train_fe = pipe.named_steps["features"].transform(X_train)
    pre = pipe.named_steps["preprocess"]
    num_cols = pre.transformers_[0][2]
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    imputed_mean = train_fe[num_cols].fillna(train_fe[num_cols].median()).mean().to_numpy()
    add("scaler_fitted_on_training_data", np.allclose(scaler.mean_, imputed_mean),
        "StandardScaler mean_ equals the training-set mean of each numeric feature")

    # 4. unseen categories / missing values at prediction time
    weird = X_test.copy()
    weird.iloc[:5, weird.columns.get_loc("country")] = "ZZ"
    weird.iloc[:5, weird.columns.get_loc("promo_code")] = "PROMO_NEVER_SEEN"
    weird.iloc[5:10, weird.columns.get_loc("age")] = np.nan
    weird.iloc[5:10, weird.columns.get_loc("device")] = np.nan
    try:
        p = pipe.predict_proba(weird)[:, 1]
        ok = np.isfinite(p).all()
    except Exception as exc:                                       # noqa: BLE001
        ok, p = False, str(exc)
    add("handles_unseen_categories_and_missing_values", ok, "predict_proba returned finite values")

    # 5. future events must not influence aggregates
    if events is not None:
        snap = pd.Timestamp(SNAPSHOT_DATE)
        ids = X_train.index.to_numpy()[:200]
        base_agg = build_aggregate_features(events, ids, snap)
        future = events.iloc[:500].copy()
        future["event_time"] = snap + pd.Timedelta(days=5)
        fut_agg = build_aggregate_features(pd.concat([events, future]), ids, snap)
        add("aggregates_ignore_events_after_snapshot", base_agg.equals(fut_agg),
            "adding events dated after the snapshot leaves aggregates unchanged")

    # 6. cross-validation isolation: the template pipeline stays unfitted, every fold is a fresh clone
    template = build_pipeline(**kw)
    cv_evaluate(template, X_train, y_train, n_splits=3, random_state=0, n_jobs=1)
    try:
        check_is_fitted(template.named_steps["features"])
        isolated = False
    except NotFittedError:
        isolated = True
    add("cv_uses_fresh_clone_per_fold", isolated, "template pipeline is still unfitted after cross-validation")

    # 7. determinism + persistence
    p1 = build_pipeline(**kw).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    p2 = build_pipeline(**kw).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    add("deterministic_refit", np.allclose(p1, p2), "two independent fits give identical predictions")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pipe.joblib"
        joblib.dump(pipe, path)
        p3 = joblib.load(path).predict_proba(X_test)[:, 1]
    add("persistence_roundtrip", np.allclose(pipe.predict_proba(X_test)[:, 1], p3),
        "joblib dump/load reproduces predictions")

    # 8. catalog covers every engineered column with the right type
    engineered = FeatureEngineer().fit(X_train).transform(X_train)
    known = set(FEATURE_CATALOG["feature"])
    types_ok = all((c in CATEGORICAL_FEATURES) == (not pd.api.types.is_numeric_dtype(engineered[c]))
                   for c in engineered.columns)
    add("feature_catalog_complete", set(engineered.columns) <= known and types_ok,
        f"{engineered.shape[1]} engineered columns documented with correct types")
    return results
