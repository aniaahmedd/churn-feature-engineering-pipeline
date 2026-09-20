"""DAY 01 - Feature engineering module.

Two pieces:

1. build_aggregate_features(events, ...)   - stateless, label-free per-entity aggregates
2. FeatureEngineer (sklearn transformer)   - numeric / date-time / interaction / categorical features
   * anything *learned* from data (median age, rare-country list, promo-code frequencies, quantile
     bin edges, price-pressure threshold) is learned in fit() -> so inside a Pipeline it is learned
     from training folds only.  y is never used.
3. CorrelationFilter                        - drops constant / near-duplicate columns (learned in fit)

Feature *groups* let Day 03 switch families on/off for ablation:
    base         always on (raw customer columns)
    datetime     features derived from signup_date
    aggregate    per-customer aggregates of the event log
    ratio        rates / ratios / logs of numeric columns
    interaction  crosses between features
    categorical  frequency-encoding, binning and rule-based states of categorical/numeric columns
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import ID_COL, SNAPSHOT_DATE

ALL_GROUPS = ("datetime", "aggregate", "ratio", "interaction", "categorical")
WINDOW_DAYS = 90

BASE_NUM = ["age", "monthly_charge", "n_support_tickets", "legacy_score"]
BASE_CAT = ["gender", "country", "plan", "acquisition_channel", "device"]
AGG_COLS = ["events_90d", "events_30d", "events_7d", "events_prior_60d", "recency_days",
            "active_days_30d", "longest_gap_days", "usage_trend_ratio", "purchase_count_90d",
            "purchase_amount_sum", "purchase_amount_mean", "purchase_amount_max", "weekend_share",
            "night_share", "login_share", "event_type_diversity"]

# --------------------------------------------------------------------------------------------
# Feature catalog: single source of truth for names, groups, types and definitions.
# Day 05 documentation is generated from this table, and tests check it covers every output.
# --------------------------------------------------------------------------------------------
_CATALOG_ROWS = [
    # name, group, type, definition
    ("age", "base", "numeric", "Customer age in years (missing values median-imputed inside the pipeline)."),
    ("monthly_charge", "base", "numeric", "Monthly subscription price."),
    ("n_support_tickets", "base", "numeric", "Number of support tickets ever opened."),
    ("legacy_score", "base", "numeric", "Score from a legacy system (kept in the baseline on purpose; expected to be noise)."),
    ("gender", "base", "categorical", "Customer gender (one-hot encoded)."),
    ("country", "base", "categorical", "Country; countries below 2% of the training rows are grouped into 'other' (learned in fit)."),
    ("plan", "base", "categorical", "Subscription plan: basic / standard / premium."),
    ("acquisition_channel", "base", "categorical", "Marketing channel through which the customer signed up."),
    ("device", "base", "categorical", "Main device; missing values become their own 'missing' level."),

    ("tenure_days", "datetime", "numeric", "Days between signup_date and the snapshot date."),
    ("tenure_log", "datetime", "numeric", "log(1 + tenure_days) - tames the right skew of tenure."),
    ("is_new_customer", "datetime", "numeric", "1 if tenure_days < 90 else 0 (onboarding-period churn is different)."),
    ("signup_month_sin", "datetime", "numeric", "sin(2*pi*signup_month/12) - cyclical encoding of signup month."),
    ("signup_month_cos", "datetime", "numeric", "cos(2*pi*signup_month/12) - cyclical encoding of signup month."),
    ("signup_is_weekend", "datetime", "numeric", "1 if the customer signed up on a Saturday or Sunday."),

    ("events_90d", "aggregate", "numeric", "Number of events (login/stream/purchase) in the 90 days before the snapshot."),
    ("events_30d", "aggregate", "numeric", "Events in the last 30 days."),
    ("events_7d", "aggregate", "numeric", "Events in the last 7 days."),
    ("events_prior_60d", "aggregate", "numeric", "Events between 90 and 30 days before the snapshot."),
    ("recency_days", "aggregate", "numeric", "Days since the most recent event (90 if no event in the window)."),
    ("active_days_30d", "aggregate", "numeric", "Distinct calendar days with at least one event in the last 30 days."),
    ("longest_gap_days", "aggregate", "numeric", "Longest inactivity gap (days) inside the 90-day window, including the gap up to the snapshot."),
    ("usage_trend_ratio", "aggregate", "numeric", "(events_30d/30) / (events_prior_60d/60 + 0.05): <1 means usage is declining."),
    ("purchase_count_90d", "aggregate", "numeric", "Number of purchase events in 90 days."),
    ("purchase_amount_sum", "aggregate", "numeric", "Total purchase amount in 90 days."),
    ("purchase_amount_mean", "aggregate", "numeric", "Mean purchase amount in 90 days (0 if none)."),
    ("purchase_amount_max", "aggregate", "numeric", "Largest single purchase in 90 days (0 if none)."),
    ("weekend_share", "aggregate", "numeric", "Share of events on Saturday/Sunday (missing if no events)."),
    ("night_share", "aggregate", "numeric", "Share of events between 00:00 and 05:59 (missing if no events)."),
    ("login_share", "aggregate", "numeric", "Share of events that are logins (missing if no events)."),
    ("event_type_diversity", "aggregate", "numeric", "Number of distinct event types used in 90 days."),

    ("tickets_per_month", "ratio", "numeric", "n_support_tickets / (tenure_days/30 + 1): ticket rate normalised by lifetime."),
    ("log_monthly_charge", "ratio", "numeric", "log(monthly_charge)."),
    ("age_missing", "ratio", "numeric", "1 if age was missing (missingness indicator)."),
    ("charge_per_event", "ratio", "numeric", "monthly_charge / (events_90d + 1): price paid per unit of usage."),
    ("log_charge_per_event", "ratio", "numeric", "log(charge_per_event + 0.01)."),
    ("events_per_tenure_day", "ratio", "numeric", "events_90d / min(tenure_days, 90): usage intensity, fair to new customers."),

    ("premium_high_tickets", "interaction", "numeric", "1 if plan == premium AND n_support_tickets >= 3 (unhappy high-value customers)."),
    ("basic_new_customer", "interaction", "numeric", "1 if plan == basic AND tenure_days < 90."),
    ("tickets_x_recency", "interaction", "numeric", "n_support_tickets * recency_days."),
    ("charge_x_recency", "interaction", "numeric", "monthly_charge * recency_days."),
    ("paid_search_low_usage", "interaction", "numeric", "1 if channel == paid_search AND usage_trend_ratio < 0.6."),

    ("promo_freq", "categorical", "numeric", "Frequency encoding of promo_code: share of training rows with that code (0 for unseen codes). No target encoding -> no leakage."),
    ("age_band", "categorical", "categorical", "Age quintile band (q1..q5, 'unknown' if missing); quintile edges learned on training data."),
    ("price_pressure", "categorical", "categorical", "'high' if charge_per_event is above the training 75th percentile, else 'normal'."),
    ("usage_state", "categorical", "categorical", "Rule-based: 'declining' if usage_trend_ratio < 0.6, 'growing' if > 1.4, else 'stable'."),
]
FEATURE_CATALOG = pd.DataFrame(_CATALOG_ROWS, columns=["feature", "group", "type", "definition"])
CATEGORICAL_FEATURES = set(FEATURE_CATALOG.loc[FEATURE_CATALOG["type"] == "categorical", "feature"])
REQUIRED_INPUT_COLUMNS = ["signup_date", "promo_code"] + BASE_NUM + BASE_CAT + AGG_COLS


# --------------------------------------------------------------------------------------------
# 1. Aggregate features from the event log (stateless, label-free, uses only events <= snapshot)
# --------------------------------------------------------------------------------------------
def build_aggregate_features(events, customer_ids, snapshot):
    ev = events.loc[events["event_time"] <= snapshot].copy()
    ev["days_ago"] = (snapshot - ev["event_time"]).dt.total_seconds() / 86400
    ev = ev.loc[ev["days_ago"] <= WINDOW_DAYS]
    ev = ev.sort_values([ID_COL, "days_ago"])
    ev["is_weekend"] = (ev["event_time"].dt.dayofweek >= 5).astype(float)
    ev["is_night"] = (ev["event_time"].dt.hour < 6).astype(float)
    ev["is_login"] = (ev["event_type"] == "login").astype(float)
    ev["day"] = ev["event_time"].dt.floor("D")
    g = ev.groupby(ID_COL)

    def count(mask):
        return ev.loc[mask].groupby(ID_COL).size()

    purchases = ev.loc[ev["event_type"] == "purchase"].groupby(ID_COL)["amount"]
    gap = ev.groupby(ID_COL)["days_ago"].diff()          # gaps between consecutive events
    longest_inner = gap.groupby(ev[ID_COL]).max()

    agg = pd.DataFrame({
        "events_90d": g.size(),
        "events_30d": count(ev["days_ago"] <= 30),
        "events_7d": count(ev["days_ago"] <= 7),
        "events_prior_60d": count(ev["days_ago"] > 30),
        "recency_days": g["days_ago"].min(),
        "active_days_30d": ev.loc[ev["days_ago"] <= 30].groupby(ID_COL)["day"].nunique(),
        "purchase_count_90d": purchases.size(),
        "purchase_amount_sum": purchases.sum(),
        "purchase_amount_mean": purchases.mean(),
        "purchase_amount_max": purchases.max(),
        "weekend_share": g["is_weekend"].mean(),
        "night_share": g["is_night"].mean(),
        "login_share": g["is_login"].mean(),
        "event_type_diversity": g["event_type"].nunique(),
    })
    agg["longest_gap_days"] = np.maximum(longest_inner.reindex(agg.index).fillna(0), agg["recency_days"])
    agg = agg.reindex(customer_ids)
    agg.index.name = ID_COL

    zero_fill = ["events_90d", "events_30d", "events_7d", "events_prior_60d", "active_days_30d",
                 "purchase_count_90d", "purchase_amount_sum", "purchase_amount_mean",
                 "purchase_amount_max", "event_type_diversity"]
    agg[zero_fill] = agg[zero_fill].fillna(0)
    agg["recency_days"] = agg["recency_days"].fillna(WINDOW_DAYS)         # domain rule: no event -> window length
    agg["longest_gap_days"] = agg["longest_gap_days"].fillna(WINDOW_DAYS)
    agg["usage_trend_ratio"] = (agg["events_30d"] / 30) / (agg["events_prior_60d"] / 60 + 0.05)
    return agg[AGG_COLS].reset_index()


# --------------------------------------------------------------------------------------------
# 2. Row-level + fitted feature transformer
# --------------------------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates engineered features.  fit() learns only from X (never y).

    Parameters
    ----------
    groups : iterable of feature groups to build on top of the always-on 'base' columns.
    drop_features : feature names to remove from the output (used after the Day 05 review).
    rare_min_freq : countries rarer than this share of training rows are grouped as 'other'.
    """

    def __init__(self, groups=ALL_GROUPS, drop_features=(), rare_min_freq=0.02, snapshot=SNAPSHOT_DATE):
        self.groups = groups
        self.drop_features = drop_features
        self.rare_min_freq = rare_min_freq
        self.snapshot = snapshot

    # ---- learned statistics (training data only) -------------------------------------------
    def fit(self, X, y=None):
        self._require(X)
        self.age_edges_ = np.unique(X["age"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())
        freq = X["country"].value_counts(normalize=True)
        self.country_keep_ = sorted(freq[freq >= self.rare_min_freq].index)
        self.promo_freq_ = X["promo_code"].value_counts(normalize=True).to_dict()
        cpe = X["monthly_charge"] / (X["events_90d"] + 1)
        self.price_pressure_threshold_ = float(cpe.quantile(0.75))
        self.feature_names_out_ = list(self._build(X).columns)
        return self

    def transform(self, X):
        self._require(X)
        return self._build(X)[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)

    # ---- internals --------------------------------------------------------------------------
    @staticmethod
    def _require(X):
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise KeyError(f"FeatureEngineer input is missing columns: {missing}")

    def _build(self, X):
        groups = set(self.groups)
        unknown = groups - set(ALL_GROUPS)
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
        snap = pd.Timestamp(self.snapshot)
        signup = pd.to_datetime(X["signup_date"])
        tenure = (snap - signup).dt.days.astype(float)
        cpe = X["monthly_charge"] / (X["events_90d"] + 1)
        out = {}

        # base (always on)
        for c in BASE_NUM:
            out[c] = X[c].astype(float)
        out["gender"] = X["gender"]
        out["country"] = X["country"].where(X["country"].isin(self.country_keep_), "other")
        out["plan"] = X["plan"]
        out["acquisition_channel"] = X["acquisition_channel"]
        out["device"] = X["device"]

        if "datetime" in groups:
            month = signup.dt.month
            out["tenure_days"] = tenure
            out["tenure_log"] = np.log1p(tenure)
            out["is_new_customer"] = (tenure < 90).astype(float)
            out["signup_month_sin"] = np.sin(2 * np.pi * month / 12)
            out["signup_month_cos"] = np.cos(2 * np.pi * month / 12)
            out["signup_is_weekend"] = (signup.dt.dayofweek >= 5).astype(float)

        if "aggregate" in groups:
            for c in AGG_COLS:
                out[c] = X[c].astype(float)

        if "ratio" in groups:
            out["tickets_per_month"] = X["n_support_tickets"] / (tenure / 30 + 1)
            out["log_monthly_charge"] = np.log(X["monthly_charge"].clip(lower=0.01))
            out["age_missing"] = X["age"].isna().astype(float)
            out["charge_per_event"] = cpe
            out["log_charge_per_event"] = np.log(cpe + 0.01)
            out["events_per_tenure_day"] = X["events_90d"] / np.minimum(tenure, WINDOW_DAYS).clip(lower=1)

        if "interaction" in groups:
            out["premium_high_tickets"] = ((X["plan"] == "premium") & (X["n_support_tickets"] >= 3)).astype(float)
            out["basic_new_customer"] = ((X["plan"] == "basic") & (tenure < 90)).astype(float)
            out["tickets_x_recency"] = X["n_support_tickets"] * X["recency_days"]
            out["charge_x_recency"] = X["monthly_charge"] * X["recency_days"]
            out["paid_search_low_usage"] = ((X["acquisition_channel"] == "paid_search")
                                            & (X["usage_trend_ratio"] < 0.6)).astype(float)

        if "categorical" in groups:
            out["promo_freq"] = X["promo_code"].map(self.promo_freq_).fillna(0.0).astype(float)
            labels = np.array([f"q{i + 1}" for i in range(len(self.age_edges_) + 1)], dtype=object)
            band = labels[np.searchsorted(self.age_edges_, X["age"].to_numpy(dtype=float))]
            out["age_band"] = pd.Series(np.where(X["age"].isna(), "unknown", band), index=X.index, dtype=object)
            out["price_pressure"] = pd.Series(np.where(cpe > self.price_pressure_threshold_, "high", "normal"),
                                              index=X.index, dtype=object)
            out["usage_state"] = pd.Series(np.select([X["usage_trend_ratio"] < 0.6, X["usage_trend_ratio"] > 1.4],
                                                     ["declining", "growing"], "stable"),
                                           index=X.index, dtype=object)

        frame = pd.DataFrame(out, index=X.index)
        order = [f for f in FEATURE_CATALOG["feature"] if f in frame.columns]
        frame = frame[order]
        return frame.drop(columns=[c for c in self.drop_features if c in frame.columns])


# --------------------------------------------------------------------------------------------
# 3. Redundancy filter (fitted on the training fold only)
# --------------------------------------------------------------------------------------------
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops constant columns and one of every pair with |Pearson r| > threshold (earlier column wins)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        df.columns = [str(c) for c in df.columns]
        self.columns_in_ = list(df.columns)
        constant = [c for c in df.columns if df[c].std(ddof=0) == 0 or df[c].nunique() <= 1]
        corr = df.drop(columns=constant).corr().abs().fillna(0).to_numpy()
        names = [c for c in df.columns if c not in constant]
        dropped = set(constant)
        for j in range(len(names)):
            if names[j] in dropped:
                continue
            for i in range(j):
                if names[i] not in dropped and corr[i, j] > self.threshold:
                    dropped.add(names[j])
                    break
        self.dropped_ = sorted(dropped)
        self.keep_ = [c for c in self.columns_in_ if c not in dropped]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        df.columns = [str(c) for c in df.columns]
        return df[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_, dtype=object)
