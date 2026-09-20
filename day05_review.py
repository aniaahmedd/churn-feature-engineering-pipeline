"""DAY 05 - Feature review, documentation, and the final validated pipeline.

Steps
  1. Permutation importance of every engineered feature on 10 validation folds (5-fold x 2 repeats).
     Each fold fits FeatureEngineer + model on its own training part only.
  2. Verdict per feature   (rules fixed in advance, PR-AUC scale):
        useful    mean importance >= 0.002 AND positive in >= 80% of folds
        drop      mean importance <  0.0005 OR positive in <  50% of folds
        unstable  everything in between (signal exists but is weak / fold-dependent)
  3. Candidate feature sets (full / drop-'drop' / only-'useful') are compared with repeated CV on the
     training split; the SMALLEST set within 0.003 PR-AUC of the best is finalised.
  4. Operating threshold is chosen from out-of-fold predictions on train (max F1).
  5. The hold-out test set is used ONCE: baseline vs full vs final, with bootstrap confidence intervals.
  6. Re-run the leakage / reproducibility checks on the final pipeline, save it, and generate
     docs/FEATURE_DOCUMENTATION.md and docs/RESULTS_SUMMARY.md.
"""
import json
import platform

import imblearn
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import DOCS_DIR, N_SPLITS, OUT_DIR, RANDOM_STATE, SNAPSHOT_DATE
from .data import load_raw, load_splits
from .evaluation import cv_evaluate, score_binary
from .features import ALL_GROUPS, FEATURE_CATALOG, FeatureEngineer
from .pipeline import build_pipeline, build_tail
from .validation import run_checks

KEEP_MEAN, DROP_MEAN, KEEP_SHARE, DROP_SHARE = 0.002, 0.0005, 0.8, 0.5
SIZE_TOLERANCE = 0.003


# ----------------------------------------------------------------------------------------------
def load_choice():
    path = OUT_DIR / "day04_choice.json"
    if path.exists():
        c = json.loads(path.read_text())
        return c["model"], c["strategy"]
    return "logreg", "none"


def permutation_review(X, y, model, strategy, n_repeats_perm=5):
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=2, random_state=RANDOM_STATE)
    per_fold = {}
    for k, (tr, va) in enumerate(splitter.split(X, y)):
        fe = FeatureEngineer(groups=ALL_GROUPS).fit(X.iloc[tr])          # learned on the training part only
        F_tr, F_va = fe.transform(X.iloc[tr]), fe.transform(X.iloc[va])
        tail = build_tail(model, strategy, random_state=RANDOM_STATE).fit(F_tr, y.iloc[tr])
        pi = permutation_importance(tail, F_va, y.iloc[va], scoring="average_precision",
                                    n_repeats=n_repeats_perm, random_state=k)
        per_fold[k] = pd.Series(pi.importances_mean, index=F_va.columns)
    return pd.DataFrame(per_fold)                                          # features x folds


def verdicts(imp):
    out = pd.DataFrame({"perm_importance_mean": imp.mean(axis=1), "perm_importance_std": imp.std(axis=1, ddof=1),
                        "share_folds_positive": (imp > 0).mean(axis=1)})
    out["verdict"] = np.select(
        [(out["perm_importance_mean"] >= KEEP_MEAN) & (out["share_folds_positive"] >= KEEP_SHARE),
         (out["perm_importance_mean"] < DROP_MEAN) | (out["share_folds_positive"] < DROP_SHARE)],
        ["useful", "drop"], "unstable")
    return out.sort_values("perm_importance_mean", ascending=False)


def bootstrap_ci(y, p_a, p_b=None, n=1000, seed=0):
    """95% bootstrap CIs for PR-AUC and ROC-AUC of p_a; if p_b is given, of the paired difference a-b."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    out = {"pr_auc": [], "roc_auc": []}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        a = score_binary(y[idx], p_a[idx])
        if p_b is None:
            for m in out:
                out[m].append(a[m])
        else:
            b = score_binary(y[idx], p_b[idx])
            for m in out:
                out[m].append(a[m] - b[m])
    return {m: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for m, v in out.items()}


def best_f1_threshold(y, proba):
    grid = np.linspace(0.03, 0.7, 136)
    f1s = [score_binary(y, proba, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def md_table(df, floatfmt=3):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)) + "\n"


# ----------------------------------------------------------------------------------------------
def main():
    X_train, X_test, y_train, y_test = load_splits()
    _, events = load_raw()
    model, strategy = load_choice()
    print(f"Configuration from Day 04: model={model}, strategy={strategy}")

    # 1-2. importance + verdicts -------------------------------------------------------------
    print("\n1) permutation importance over 10 validation folds ...")
    imp = permutation_review(X_train, y_train, model, strategy)
    rank_corr = imp.corr(method="spearman").to_numpy()
    mean_rank_corr = float(rank_corr[np.triu_indices_from(rank_corr, k=1)].mean())
    review = verdicts(imp)
    day01 = pd.read_csv(OUT_DIR / "day01_feature_summary.csv").set_index("feature")
    review = FEATURE_CATALOG.set_index("feature").join(review).join(day01[["mutual_info", "redundant_corr_gt_0.95"]])
    review["verdict"] = review["verdict"].fillna("drop")
    review = review.sort_values("perm_importance_mean", ascending=False)
    review.reset_index().to_csv(OUT_DIR / "day05_feature_review.csv", index=False)
    print(review["verdict"].value_counts().to_string())
    print(f"   mean pairwise Spearman rank-correlation of importances between folds: {mean_rank_corr:.2f}")
    print(review[["group", "perm_importance_mean", "perm_importance_std", "share_folds_positive", "verdict"]]
          .head(15).round(4).to_string())

    # 3. candidate feature sets -----------------------------------------------------------------
    print("\n2) candidate feature sets (5-fold x 3 repeats on train)")
    all_feats = list(review.index)
    variants = {
        "full": [],
        "prune_drop": [f for f in all_feats if review.loc[f, "verdict"] == "drop"],
        "only_useful": [f for f in all_feats if review.loc[f, "verdict"] != "useful"],
    }
    rows = []
    for name, dropped in variants.items():
        pipe = build_pipeline(model=model, strategy=strategy, drop_features=dropped, random_state=RANDOM_STATE)
        folds, _ = cv_evaluate(pipe, X_train, y_train, n_repeats=3, random_state=123)
        rows.append({"variant": name, "n_engineered_features": len(all_feats) - len(dropped),
                     "pr_auc_mean": folds["pr_auc"].mean(), "pr_auc_std": folds["pr_auc"].std(ddof=1),
                     "roc_auc_mean": folds["roc_auc"].mean(), "f1_mean": folds["f1"].mean()})
    cand = pd.DataFrame(rows)
    best = cand["pr_auc_mean"].max()
    ok = cand[cand["pr_auc_mean"] >= best - SIZE_TOLERANCE].sort_values("n_engineered_features")
    final_variant = ok.iloc[0]["variant"]
    cand["chosen"] = cand["variant"] == final_variant
    cand.to_csv(OUT_DIR / "day05_candidate_sets.csv", index=False)
    print(cand.round(3).to_string(index=False))
    print(f"   -> final feature set: {final_variant}")
    drop_features = variants[final_variant]
    kept_features = [f for f in all_feats if f not in drop_features]

    # 4. threshold from OOF predictions ---------------------------------------------------------
    final_kwargs = dict(model=model, strategy=strategy, drop_features=tuple(drop_features), random_state=RANDOM_STATE)
    _, oof = cv_evaluate(build_pipeline(**final_kwargs), X_train, y_train, n_repeats=1, random_state=2024)
    threshold = best_f1_threshold(y_train, oof[0])
    print(f"\n3) operating threshold chosen on out-of-fold train predictions: {threshold:.2f}")

    # 5. hold-out test, used once ---------------------------------------------------------------
    print("\n4) hold-out test set (used once)")
    fitted = {}
    for name, kw in {"baseline": dict(groups=(), model=model, strategy="none"),
                     "full_engineered": dict(model=model, strategy=strategy),
                     "final": final_kwargs}.items():
        fitted[name] = build_pipeline(**kw).fit(X_train, y_train)
    proba = {n: p.predict_proba(X_test)[:, 1] for n, p in fitted.items()}
    test_rows = []
    for name, p in proba.items():
        ci = bootstrap_ci(y_test, p)
        for thr_name, thr in (("0.5", 0.5), (f"tuned ({threshold:.2f})", threshold)):
            s = score_binary(y_test, p, thr)
            test_rows.append({"pipeline": name, "threshold": thr_name, **s,
                              "pr_auc_ci95": f"{ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f}",
                              "roc_auc_ci95": f"{ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}"})
    test = pd.DataFrame(test_rows)
    test.to_csv(OUT_DIR / "day05_test_results.csv", index=False)
    diff = bootstrap_ci(y_test, proba["final"], proba["baseline"])
    print(test.round(3).to_string(index=False))
    print(f"   final - baseline paired bootstrap: PR-AUC {diff['pr_auc'][0]:+.3f}..{diff['pr_auc'][1]:+.3f}, "
          f"ROC-AUC {diff['roc_auc'][0]:+.3f}..{diff['roc_auc'][1]:+.3f}")
    print(f"   test prevalence = {y_test.mean():.3f}")

    # 6. validate + save ----------------------------------------------------------------------------
    print("\n5) final leakage / reproducibility checks")
    checks = pd.DataFrame(run_checks(X_train, y_train, X_test, events=events, pipeline_kwargs=final_kwargs))
    checks.to_csv(OUT_DIR / "day05_final_checks.csv", index=False)
    print(checks[["check", "passed"]].to_string(index=False))
    assert checks["passed"].all(), "final pipeline failed a validation check"

    joblib.dump(fitted["final"], OUT_DIR / "final_pipeline.joblib")
    fe_fit = fitted["final"].named_steps["features"]
    learned = {"country_keep": list(fe_fit.country_keep_), "age_quintile_edges": [float(v) for v in fe_fit.age_edges_],
               "price_pressure_threshold": fe_fit.price_pressure_threshold_,
               "n_promo_codes_seen": len(fe_fit.promo_freq_),
               "correlation_filter_dropped": list(fitted["final"].named_steps["correlation_filter"].dropped_)}
    config = {"model": model, "imbalance_strategy": strategy, "feature_groups": list(ALL_GROUPS),
              "dropped_features": drop_features, "n_engineered_features": len(kept_features),
              "operating_threshold": threshold, "snapshot_date": SNAPSHOT_DATE, "random_state": RANDOM_STATE,
              "n_train": len(X_train), "n_test": len(X_test),
              "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                           "pandas": pd.__version__, "numpy": np.__version__, "imbalanced-learn": imblearn.__version__},
              "learned_from_train": learned}
    (OUT_DIR / "final_pipeline_config.json").write_text(json.dumps(config, indent=2, default=str))
    (OUT_DIR / "selected_features.json").write_text(json.dumps(
        {"kept": kept_features, "dropped": drop_features,
         "useful": review.index[review["verdict"] == "useful"].tolist(),
         "unstable": review.index[review["verdict"] == "unstable"].tolist()}, indent=2))

    # ---- plots -------------------------------------------------------------------------------------
    top = review.head(20).iloc[::-1]
    colors = top["verdict"].map({"useful": "#3b7dd8", "unstable": "#e0a030", "drop": "#bbbbbb"})
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top["perm_importance_mean"], xerr=top["perm_importance_std"], color=colors, capsize=2)
    ax.set_xlabel("permutation importance (drop in PR-AUC), mean +/- std over 10 folds")
    ax.set_title("Day 05 - top 20 engineered features (blue=useful, orange=unstable, grey=drop)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "day05_permutation_importance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, style in (("baseline", "--"), ("full_engineered", ":"), ("final", "-")):
        pr, rc, _ = precision_recall_curve(y_test, proba[name])
        ax.plot(rc, pr, style, label=name)
    ax.axhline(y_test.mean(), color="grey", lw=0.6)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Day 05 - hold-out precision-recall curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "day05_test_pr_curve.png", dpi=130)
    plt.close(fig)

    write_docs(review, cand, final_variant, test, diff, checks, config, mean_rank_corr, threshold)
    print("\nDay 05 done -> docs/FEATURE_DOCUMENTATION.md, docs/RESULTS_SUMMARY.md, outputs/final_pipeline.joblib")


# ----------------------------------------------------------------------------------------------
def write_docs(review, cand, final_variant, test, diff, checks, config, mean_rank_corr, threshold):
    r = review.reset_index()
    useful = r[r["verdict"] == "useful"]["feature"].tolist()
    unstable = r[r["verdict"] == "unstable"]["feature"].tolist()
    dropped = r[r["verdict"] == "drop"]["feature"].tolist()
    L = config["learned_from_train"]
    at_half = test[test["threshold"] == "0.5"].set_index("pipeline")["pr_auc"]
    hold_full, hold_final = at_half["full_engineered"], at_half["final"]

    doc = [f"""# Feature documentation

Generated by `src/day05_review.py`. Regenerate with `python run_all.py --day 5`.

## 1. Purpose and scope
Features for predicting **`churn_30d`** (customer churns within 30 days after the snapshot date
{config['snapshot_date']}) for a subscription service. Positive class is rare (~10.5%), so **PR-AUC** is the
headline metric; ROC-AUC and threshold metrics are secondary.

Final configuration: model = `{config['model']}`, imbalance strategy = `{config['imbalance_strategy']}`,
feature set = `{final_variant}` ({config['n_engineered_features']} engineered features kept, {len(config['dropped_features'])} dropped),
operating threshold = {threshold:.2f} (chosen on out-of-fold train predictions).

## 2. Pipeline (all learning happens on training data only)
```
raw model frame (customers + event aggregates)
   -> FeatureEngineer            learns: rare-country list, promo-code frequencies, age quintile edges,
                                         price-pressure threshold          (unsupervised, never sees y)
   -> ColumnTransformer          numeric: median impute -> StandardScaler
                                 categorical: 'missing' impute -> OneHotEncoder(handle_unknown, min_frequency=1%)
   -> CorrelationFilter          drops constant columns and one of each pair with |r| > 0.95
   -> [resampler]                only if the chosen strategy needs one (fit-time only)
   -> model
```
* Everything above is one `imblearn.pipeline.Pipeline`; cross-validation clones it for every fold, so no
  statistic is ever shared between training and validation data.
* Event aggregates (`build_aggregate_features`) are per-customer, label-free and only use events dated on
  or before the snapshot; they are computed before the split because they cannot leak information across rows.
* No target encoding is used anywhere (categorical signal is captured by one-hot, frequency encoding and binning).
* Automated checks ({int(checks['passed'].sum())}/{len(checks)} passing): {', '.join('`' + c.split(':')[0] + '`' for c in checks['check'])}.

### Values learned from the training split (final pipeline)
* countries kept (others -> `other`): {', '.join(L['country_keep'])}
* age quintile edges: {', '.join(f'{v:.0f}' for v in L['age_quintile_edges'])}
* price-pressure threshold (75th percentile of `charge_per_event`): {L['price_pressure_threshold']:.3f}
* promo codes with a learned frequency: {L['n_promo_codes_seen']}
* columns removed by the correlation filter: {', '.join(L['correlation_filter_dropped']) or 'none'}

## 3. Review outcome
Permutation importance (drop in PR-AUC when a feature is shuffled) on 10 validation folds; each fold fits its own
feature engineer and model. Mean pairwise Spearman correlation of the importance rankings between folds: **{mean_rank_corr:.2f}**.

* **Useful ({len(useful)}):** {', '.join(f'`{f}`' for f in useful) or 'none'}
* **Unstable / weak ({len(unstable)}):** {', '.join(f'`{f}`' for f in unstable) or 'none'}
* **No signal - candidates to drop ({len(dropped)}):** {', '.join(f'`{f}`' for f in dropped) or 'none'}

Rules (fixed before looking): useful = mean >= {KEEP_MEAN} and positive in >= {int(KEEP_SHARE * 100)}% of folds;
drop = mean < {DROP_MEAN} or positive in < {int(DROP_SHARE * 100)}% of folds; otherwise unstable.

Candidate feature sets compared with 5-fold x 3-repeat CV (smallest set within {SIZE_TOLERANCE} PR-AUC of the best is chosen):

{md_table(cand)}
> Caveat: permutation importance under-rates features that have correlated twins (for example `recency_days`,
> `longest_gap_days`, `active_days_30d`, `events_30d` all describe recent activity), which is why the pruned
> sets are validated by re-running cross-validation instead of trusting the verdict alone.

## 4. Feature dictionary
"""]
    for group in ["base", "datetime", "aggregate", "ratio", "interaction", "categorical"]:
        sub = r[r["group"] == group][["feature", "type", "definition", "verdict", "perm_importance_mean",
                                      "share_folds_positive", "mutual_info"]].copy()
        sub.columns = ["feature", "type", "definition", "verdict", "perm. importance", "folds > 0", "mutual info"]
        doc.append(f"### {group}\n\n{md_table(sub, 4)}\n")
    doc.append("""## 5. Reproducing
```bash
pip install -r requirements.txt
python run_all.py            # data -> Day 1..5 -> docs + final pipeline
pytest -q                    # leakage / reproducibility tests
```
Using the saved pipeline:
```python
import joblib, pandas as pd
from src.data import load_model_frame
pipe = joblib.load("outputs/final_pipeline.joblib")
X, y = load_model_frame()
churn_probability = pipe.predict_proba(X)[:, 1]
```
Versions, seeds and learned parameters are stored in `outputs/final_pipeline_config.json`.

## 6. Known limitations
* Data are synthetic: numbers show the *method*, not real-world performance. Replace `src/data.py` to use real data.
* Feature groups overlap in information (ratios, interactions and encodings are built from the same raw signals), so
  single-group ablations understate value; see the add-on-aggregate experiment in RESULTS_SUMMARY.md.
* Importance-based pruning used the same training folds as model selection, so the CV gain from pruning is optimistic.
  On the untouched hold-out set (threshold 0.5) PR-AUC is """ + f"{hold_full:.3f} for all features vs {hold_final:.3f} for the final set" + """:
  the pruned pipeline is smaller and simpler, with equal accuracy within noise (the hold-out has only about 130 positives).
""")
    (DOCS_DIR / "FEATURE_DOCUMENTATION.md").write_text("\n".join(doc), encoding="utf-8")

    # ---- results summary -------------------------------------------------------------------------
    abl = pd.read_csv(OUT_DIR / "day03_ablation_results.csv")
    strat = pd.read_csv(OUT_DIR / "day04_strategy_results.csv")
    seg = pd.read_csv(OUT_DIR / "day04_segment_results.csv")
    imb = pd.read_csv(OUT_DIR / "day04_imbalance_summary.csv")
    d2 = pd.read_csv(OUT_DIR / "day02_pipeline_cv.csv")
    t = test.copy()
    summary = [f"""# Results summary (Days 1-5)

All experiment numbers except the hold-out table come from cross-validation on the 80% training split.

## Day 02 - pipeline sanity check
{md_table(d2)}
## Day 03 - baseline vs engineered features (5-fold x 2 repeats, paired folds)
`delta_pr_auc` is against the reference (baseline for add_one, full for drop_one, baseline+aggregate for add_on_aggregate).

{md_table(abl[['model', 'experiment', 'group', 'pr_auc_mean', 'pr_auc_std', 'roc_auc_mean', 'delta_pr_auc', 'delta_ci95_low', 'delta_ci95_high', 'folds_better_than_ref']])}
## Day 04 - imbalance
{md_table(imb[imb['scope'].isin(['train', 'hold-out'])][['scope', 'n', 'positives', 'prevalence']], 4)}
Strategies (5-fold x 3 seeds; resampling happens inside training folds only):

{md_table(strat[['model', 'strategy', 'pr_auc_mean', 'pr_auc_fold_std', 'pr_auc_seed_std', 'roc_auc_mean', 'f1_mean', 'recall_mean', 'precision_mean']])}
Segments (out-of-fold, chosen configuration; `low_support` = fewer than 30 positives, treat with caution):

{md_table(seg[['scope', 'segment', 'n', 'positives', 'pr_auc_baseline', 'pr_auc_engineered', 'delta_pr_auc', 'pr_auc_seed_std', 'low_support']])}
## Day 05 - hold-out test set (used once)
{md_table(t)}
Paired bootstrap, final minus baseline: PR-AUC {diff['pr_auc'][0]:+.3f} to {diff['pr_auc'][1]:+.3f}; ROC-AUC {diff['roc_auc'][0]:+.3f} to {diff['roc_auc'][1]:+.3f} (95% CI).
"""]
    (DOCS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
