"""Data access layer.  To use your own dataset, edit this file only:
   * load_raw()            -> customers table (one row per entity) + event log
   * load_model_frame()    -> X (features indexed by entity id) and y
Everything downstream (features, pipeline, experiments) only sees X and y.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, ID_COL, RANDOM_STATE, SNAPSHOT_DATE, TARGET, TEST_SIZE
from .features import build_aggregate_features


def load_raw():
    """Read the CSVs (generating them first if they do not exist yet)."""
    if not (DATA_DIR / "customers.csv").exists():
        from .generate_data import generate
        cust, ev = generate()
        cust.to_csv(DATA_DIR / "customers.csv", index=False)
        ev.to_csv(DATA_DIR / "events.csv", index=False)
    customers = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])
    events = pd.read_csv(DATA_DIR / "events.csv", parse_dates=["event_time"])
    return customers, events


def load_model_frame():
    """Customers + per-customer event aggregates (computed strictly from events <= snapshot).

    Aggregates are per-entity and label-free, so computing them once before the split does not
    leak anything.  Anything that is *learned across rows* (medians, category frequencies,
    scaling, thresholds) lives inside the sklearn Pipeline and is fitted on training folds only.
    """
    customers, events = load_raw()
    agg = build_aggregate_features(events, customers[ID_COL].to_numpy(), pd.Timestamp(SNAPSHOT_DATE))
    df = customers.merge(agg, on=ID_COL, how="left").set_index(ID_COL)
    y = df.pop(TARGET).astype(int)
    return df, y


def load_splits():
    """Stratified train / hold-out split.  The hold-out set is used once, on Day 05."""
    X, y = load_model_frame()
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def add_segments(X):
    """Segment labels used for robustness checks (Day 04)."""
    seg = pd.DataFrame(index=X.index)
    tenure = (pd.Timestamp(SNAPSHOT_DATE) - pd.to_datetime(X["signup_date"])).dt.days
    seg["tenure_bucket"] = np.select([tenure < 90, tenure < 365], ["new (<90d)", "mid (90-365d)"], "long (365d+)")
    for c in ("plan", "country", "acquisition_channel", "device"):
        seg[c] = X[c].fillna("missing")
    return seg
