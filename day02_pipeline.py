"""DAY 02 - Data transformation pipeline (encoding, scaling, leakage-safe).

Output: updated ML pipeline (src/pipeline.py) + evidence that it is leakage-safe
    outputs/day02_pipeline.html            visual diagram of the pipeline (open in a browser)
    outputs/day02_leakage_checks.csv       automated checks (all must pass)
    outputs/day02_pipeline_cv.csv          baseline-vs-full sanity CV scores
    outputs/day02_pipeline_train_fit.joblib  pipeline fitted on the training split
"""
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.utils import estimator_html_repr

from .config import OUT_DIR
from .data import load_raw, load_splits
from .evaluation import cv_evaluate, summarize
from .pipeline import build_pipeline
from .validation import run_checks


def main():
    X_train, X_test, y_train, y_test = load_splits()
    _, events = load_raw()

    pipe = build_pipeline()
    (OUT_DIR / "day02_pipeline.html").write_text(estimator_html_repr(pipe), encoding="utf-8")

    checks = pd.DataFrame(run_checks(X_train, y_train, X_test, events=events))
    checks.to_csv(OUT_DIR / "day02_leakage_checks.csv", index=False)
    print(checks.to_string(index=False))
    assert checks["passed"].all(), "a leakage / robustness check failed - see day02_leakage_checks.csv"

    # sanity CV: a dummy model, the raw-columns pipeline, and the full pipeline (5-fold, train split only)
    rows = []
    dummy = DummyClassifier(strategy="prior")
    for name, model in [("dummy (prevalence)", dummy),
                        ("pipeline: base columns only", build_pipeline(groups=())),
                        ("pipeline: all engineered features", build_pipeline())]:
        folds, _ = cv_evaluate(model, X_train, y_train, random_state=0)
        rows.append({"model": name, **summarize(folds, ["pr_auc", "roc_auc", "f1"])})
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT_DIR / "day02_pipeline_cv.csv", index=False)
    print("\n", cv.round(3).to_string(index=False))

    joblib.dump(pipe.fit(X_train, y_train), OUT_DIR / "day02_pipeline_train_fit.joblib")
    print("\nDay 02 done: all leakage checks passed.")


if __name__ == "__main__":
    main()
