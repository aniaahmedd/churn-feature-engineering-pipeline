"""Synthetic subscription-churn dataset (customers table + event log).

Why synthetic?  The brief did not fix a dataset, so this generator gives a dataset that has
everything the 5-day plan needs: numeric + categorical + date columns, an event log to
aggregate, missing values, a high-cardinality noise column, a pure-noise column, and a
class imbalance (~10-12% churners).  Swap in your own data by editing src/data.py only.

Run:  python -m src.generate_data
"""
import numpy as np
import pandas as pd

from .config import DATA_DIR, RANDOM_STATE, SNAPSHOT_DATE, TARGET, ID_COL

WINDOW_DAYS = 90


def _make_customers(n, rng, snapshot):
    tenure_days = rng.integers(20, 1400, n)
    plan = rng.choice(["basic", "standard", "premium"], n, p=[0.50, 0.35, 0.15])
    charge = pd.Series(plan).map({"basic": 9.0, "standard": 19.0, "premium": 39.0}).to_numpy()
    charge = np.round(charge * rng.normal(1.0, 0.08, n), 2)

    df = pd.DataFrame({
        ID_COL: np.arange(1, n + 1),
        "signup_date": snapshot - pd.to_timedelta(tenure_days, unit="D"),
        "age": rng.normal(37, 11, n).clip(18, 80).round(),
        "gender": rng.choice(["F", "M", "Other"], n, p=[0.48, 0.48, 0.04]),
        "country": rng.choice(["PK", "US", "UK", "AE", "DE", "CA", "IN", "SE", "NZ"], n,
                              p=[0.30, 0.20, 0.10, 0.10, 0.08, 0.07, 0.10, 0.03, 0.02]),
        "plan": plan,
        "monthly_charge": charge,
        "acquisition_channel": rng.choice(["organic", "paid_search", "social", "referral", "partner"], n,
                                          p=[0.30, 0.25, 0.20, 0.15, 0.10]),
        "device": rng.choice(["android", "ios", "web"], n, p=[0.45, 0.35, 0.20]),
        "n_support_tickets": rng.poisson(1.2, n),
        # legacy score from an old system: pure noise (a "trap" feature for Day 05 to catch)
        "legacy_score": rng.normal(50, 15, n).round(1),
        # high-cardinality categorical (~150 codes, heavy tail) with no real signal
        "promo_code": ["PROMO_" + str(k) for k in np.minimum(rng.zipf(1.6, n), 150)],
    })
    df.loc[rng.random(n) < 0.03, "age"] = np.nan
    df.loc[rng.random(n) < 0.05, "device"] = np.nan
    return df, tenure_days


def _make_events(customers, tenure_days, rng, snapshot):
    n = len(customers)
    base = rng.lognormal(np.log(0.30), 0.6, n)            # events per day
    trend = np.clip(rng.normal(1.0, 0.40, n), 0.05, 2.0)  # <1 = usage declining recently
    recent_win = np.minimum(30, tenure_days)
    prior_win = np.clip(tenure_days - 30, 0, WINDOW_DAYS - 30)
    n_recent = rng.poisson(base * trend * recent_win)
    n_prior = rng.poisson(base * prior_win)

    cust_rep = np.concatenate([np.repeat(np.arange(n), n_recent), np.repeat(np.arange(n), n_prior)])
    days_ago = np.concatenate([
        rng.random(n_recent.sum()) * np.repeat(recent_win, n_recent),
        30 + rng.random(n_prior.sum()) * np.repeat(prior_win, n_prior),
    ])
    m = len(cust_rep)
    weekend_pref = np.clip(rng.normal(0.28, 0.08, n), 0.05, 0.6)
    night_pref = np.clip(rng.normal(0.15, 0.08, n), 0.0, 0.5)
    event_type = rng.choice(["login", "stream", "purchase"], m, p=[0.45, 0.45, 0.10])
    ts = pd.Series(snapshot - pd.to_timedelta(days_ago, unit="D"))
    hour = np.where(rng.random(m) < night_pref[cust_rep], rng.integers(0, 6, m), rng.integers(6, 24, m))
    ts = ts.dt.floor("D") + pd.to_timedelta(hour, unit="h")
    dow = ts.dt.dayofweek.to_numpy()
    flip = (rng.random(m) < weekend_pref[cust_rep]) & (dow < 5)
    ts = ts + pd.to_timedelta(np.where(flip, 5 - dow, 0), unit="D")
    ts = ts.where(ts <= snapshot, snapshot - pd.Timedelta(hours=1))
    amount = np.where(event_type == "purchase", np.round(rng.lognormal(2.6, 0.7, m), 2), 0.0)
    ev = pd.DataFrame({ID_COL: customers[ID_COL].to_numpy()[cust_rep], "event_time": ts.to_numpy(),
                       "event_type": event_type, "amount": amount})
    return ev.sort_values([ID_COL, "event_time"]).reset_index(drop=True)


def _churn_labels(customers, events, tenure_days, rng, snapshot, target_rate=0.11):
    days_ago = (snapshot - events["event_time"]).dt.total_seconds() / 86400
    ev = events.assign(days_ago=days_ago)
    ids = customers[ID_COL]
    n30 = ev[ev.days_ago <= 30].groupby(ID_COL).size().reindex(ids, fill_value=0).to_numpy()
    n_prior = ev[ev.days_ago > 30].groupby(ID_COL).size().reindex(ids, fill_value=0).to_numpy()
    recency = ev.groupby(ID_COL).days_ago.min().reindex(ids).fillna(90).to_numpy()
    total = n30 + n_prior
    trend_ratio = (n30 / 30) / (n_prior / 60 + 0.05)
    plan = customers["plan"].to_numpy()
    tickets = customers["n_support_tickets"].to_numpy()

    z = (0.07 * recency
         - 0.9 * np.log(trend_ratio + 0.15)
         + 0.22 * tickets
         + 0.8 * (plan == "basic")
         + 1.1 * ((plan == "premium") & (tickets >= 3))          # interaction effect
         + 0.8 * (tenure_days < 90)
         + 0.20 * np.log(customers["monthly_charge"].to_numpy() / (total + 1) + 0.1)  # price per usage
         + 0.45 * (customers["acquisition_channel"].to_numpy() == "paid_search")
         + rng.normal(0, 0.6, len(customers)))
    lo, hi = -20.0, 20.0
    for _ in range(60):                                           # calibrate intercept to target rate
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(z + mid)))
        lo, hi = (mid, hi) if p.mean() < target_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(z + mid)))
    return (rng.random(len(customers)) < p).astype(int)


def generate(n_customers=6000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    customers, tenure_days = _make_customers(n_customers, rng, snapshot)
    events = _make_events(customers, tenure_days, rng, snapshot)
    customers[TARGET] = _churn_labels(customers, events, tenure_days, rng, snapshot)
    return customers, events


if __name__ == "__main__":
    cust, ev = generate()
    cust.to_csv(DATA_DIR / "customers.csv", index=False)
    ev.to_csv(DATA_DIR / "events.csv", index=False)
    print(f"customers: {cust.shape}, events: {ev.shape}, churn rate: {cust[TARGET].mean():.3f}")
