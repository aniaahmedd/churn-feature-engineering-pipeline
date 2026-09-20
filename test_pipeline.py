"""Leakage / robustness tests.  Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.data import load_raw, load_splits
from src.evaluation import cv_evaluate
from src.features import ALL_GROUPS, FEATURE_CATALOG, FeatureEngineer
from src.pipeline import MODELS, STRATEGIES, build_pipeline
from src.validation import run_checks


@pytest.fixture(scope="module")
def splits():
    return load_splits()


@pytest.fixture(scope="module")
def events():
    return load_raw()[1]


def test_all_validation_checks_pass(splits, events):
    Xtr, Xte, ytr, _ = splits
    results = run_checks(Xtr, ytr, Xte, events=events)
    failed = [r for r in results if not r["passed"]]
    assert not failed, failed


def test_catalog_covers_every_group_and_output(splits):
    Xtr, _, _, _ = splits
    out = FeatureEngineer().fit(Xtr).transform(Xtr)
    assert set(out.columns) <= set(FEATURE_CATALOG["feature"])
    assert set(FEATURE_CATALOG["group"]) == {"base", *ALL_GROUPS}


@pytest.mark.parametrize("groups", [(), ("datetime",), ("aggregate",), ("ratio",), ("interaction",), ("categorical",), ALL_GROUPS])
def test_every_group_combination_builds(splits, groups):
    Xtr, Xte, _, _ = splits
    fe = FeatureEngineer(groups=groups).fit(Xtr)
    assert len(fe.transform(Xte)) == len(Xte)


def test_drop_features_removes_columns(splits):
    Xtr, _, _, _ = splits
    out = FeatureEngineer(drop_features=("legacy_score", "promo_freq")).fit(Xtr).transform(Xtr)
    assert "legacy_score" not in out and "promo_freq" not in out


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pipeline_variants_train_and_predict(splits, model, strategy):
    Xtr, Xte, ytr, _ = splits
    small = Xtr.iloc[:1200]
    pipe = build_pipeline(model=model, strategy=strategy).fit(small, ytr.loc[small.index])
    p = pipe.predict_proba(Xte)[:, 1]
    assert p.shape == (len(Xte),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_cv_is_paired_and_reproducible(splits):
    Xtr, _, ytr, _ = splits
    a, _ = cv_evaluate(build_pipeline(), Xtr, ytr, n_splits=3, random_state=7, n_jobs=1)
    b, _ = cv_evaluate(build_pipeline(), Xtr, ytr, n_splits=3, random_state=7, n_jobs=1)
    pd.testing.assert_frame_equal(a, b)
