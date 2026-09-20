"""DAY 04 - Class imbalance and robustness.

1. Imbalance check      : prevalence overall / train / hold-out / per CV fold / per segment.
2. Imbalance handling   : none vs class weights vs random over-sampling vs random under-sampling vs SMOTE.
                          Resamplers live INSIDE the pipeline, so they only ever touch training folds.
3. Robustness           : 5-fold CV repeated with 3 different seeds (different folds AND model seeds);
                          we report fold-to-fold and seed-to-seed spread.
4. Segment robustness   : out-of-fold performance per plan / country / tenure / channel / device,
                          engineered features vs baseline.

Selection rule (written down before looking at results): the primary metric is PR-AUC. All
(model, strategy) combinations within 0.005 PR-AUC of the best are treated as tied; among the tied ones
we pick the highest F1 at the default 0.5 threshold.

Output: outputs/day04_*.csv, day04_strategies.png, day04_segments.png, day04_choice.json
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import N_SPLITS, OUT_DIR, ROBUSTNESS_SEEDS
from .data import add_segments, load_splits
from .evaluation import cv_evaluate, score_binary
from .features import ALL_GROUPS
from .pipeline import MODELS, STRATEGIES, build_pipeline
from sklearn.model_selection import StratifiedKFold

TIE_TOLERANCE = 0.005
LOW_SUPPORT_POSITIVES = 30


def imbalance_summary(X_train, X_test, y_train, y_test):
    rows = [{"scope": "train", "segment": "-", "n": len(y_train), "positives": int(y_train.sum()),
             "prevalence": y_train.mean()},
            {"scope": "hold-out", "segment": "-", "n": len(y_test), "positives": int(y_test.sum()),
             "prevalence": y_test.mean()}]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=ROBUSTNESS_SEEDS[0])
    for k, (_, va) in enumerate(skf.split(X_train, y_train)):
        rows.append({"scope": "cv_fold", "segment": f"fold {k}", "n": len(va), "positives": int(y_train.iloc[va].sum()),
                     "prevalence": y_train.iloc[va].mean()})
    seg = add_segments(X_train)
    for col in seg.columns:
        for level, idx in seg.groupby(col).groups.items():
            yy = y_train.loc[idx]
            rows.append({"scope": f"segment:{col}", "segment": level, "n": len(yy), "positives": int(yy.sum()),
                         "prevalence": yy.mean()})
    return pd.DataFrame(rows)


def strategy_experiments(X, y):
    rows = []
    for model in MODELS:
        for strategy in STRATEGIES:
            per_fold = []
            for seed in ROBUSTNESS_SEEDS:
                pipe = build_pipeline(model=model, strategy=strategy, random_state=seed)
                folds, _ = cv_evaluate(pipe, X, y, random_state=seed)
                folds["seed"] = seed
                per_fold.append(folds)
            f = pd.concat(per_fold, ignore_index=True)
            seed_means = f.groupby("seed")["pr_auc"].mean()
            rows.append({"model": model, "strategy": strategy,
                         **{f"{m}_mean": f[m].mean() for m in ("pr_auc", "roc_auc", "f1", "recall", "precision", "balanced_acc")},
                         "pr_auc_fold_std": f["pr_auc"].std(ddof=1), "pr_auc_seed_std": seed_means.std(ddof=1),
                         "pr_auc_min_fold": f["pr_auc"].min(), "pr_auc_max_fold": f["pr_auc"].max(),
                         "roc_auc_fold_std": f["roc_auc"].std(ddof=1), "f1_fold_std": f["f1"].std(ddof=1)})
            print(f"  {model:6s} {strategy:12s} PR-AUC {rows[-1]['pr_auc_mean']:.3f}  ROC {rows[-1]['roc_auc_mean']:.3f}  "
                  f"F1@0.5 {rows[-1]['f1_mean']:.3f}  recall {rows[-1]['recall_mean']:.3f}  seed-std {rows[-1]['pr_auc_seed_std']:.4f}")
    return pd.DataFrame(rows)


def choose(results):
    best = results["pr_auc_mean"].max()
    tied = results[results["pr_auc_mean"] >= best - TIE_TOLERANCE]
    pick = tied.sort_values(["f1_mean", "pr_auc_mean"], ascending=False).iloc[0]
    return {"model": pick["model"], "strategy": pick["strategy"], "pr_auc_mean": float(pick["pr_auc_mean"]),
            "tied_candidates": [f"{r.model}/{r.strategy}" for r in tied.itertuples()],
            "rule": f"highest F1@0.5 among combinations within {TIE_TOLERANCE} PR-AUC of the best"}


def segment_experiments(X, y, choice):
    seg = add_segments(X)
    rows = []
    for name, groups, strategy in (("engineered", ALL_GROUPS, choice["strategy"]), ("baseline", (), "none")):
        for seed in ROBUSTNESS_SEEDS:
            pipe = build_pipeline(groups=groups, model=choice["model"], strategy=strategy, random_state=seed)
            _, oof = cv_evaluate(pipe, X, y, random_state=seed)
            proba = oof[0]
            overall = score_binary(y, proba)
            rows.append({"features": name, "seed": seed, "scope": "overall", "segment": "all", "n": len(y),
                         "positives": int(y.sum()), **overall})
            for col in seg.columns:
                for level, idx in seg.groupby(col).groups.items():
                    mask = X.index.isin(idx)
                    if y[mask].nunique() < 2:
                        continue
                    rows.append({"features": name, "seed": seed, "scope": col, "segment": level, "n": int(mask.sum()),
                                 "positives": int(y[mask].sum()), **score_binary(y[mask], proba[mask])})
    raw = pd.DataFrame(rows)
    agg = (raw.groupby(["features", "scope", "segment"])
              .agg(n=("n", "first"), positives=("positives", "first"),
                   pr_auc=("pr_auc", "mean"), pr_auc_seed_std=("pr_auc", "std"),
                   roc_auc=("roc_auc", "mean"), recall=("recall", "mean")).reset_index())
    agg["prevalence"] = agg["positives"] / agg["n"]
    agg["lift_over_prevalence"] = agg["pr_auc"] / agg["prevalence"]
    agg["low_support"] = agg["positives"] < LOW_SUPPORT_POSITIVES
    wide = agg.pivot(index=["scope", "segment", "n", "positives", "prevalence", "low_support"],
                     columns="features", values=["pr_auc", "roc_auc", "lift_over_prevalence"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["delta_pr_auc"] = wide["pr_auc_engineered"] - wide["pr_auc_baseline"]
    seed_std = agg[agg["features"] == "engineered"][["scope", "segment", "pr_auc_seed_std"]]
    wide = wide.merge(seed_std, on=["scope", "segment"], how="left")
    return wide.sort_values(["scope", "segment"]).reset_index(drop=True)


def main():
    X_train, X_test, y_train, y_test = load_splits()

    print("1) class imbalance")
    imb = imbalance_summary(X_train, X_test, y_train, y_test)
    imb.to_csv(OUT_DIR / "day04_imbalance_summary.csv", index=False)
    print(imb[imb["scope"].isin(["train", "hold-out"])].round(4).to_string(index=False))
    print(f"   imbalance ratio (negatives : positives) in train = {(1 - y_train.mean()) / y_train.mean():.1f} : 1")

    print("\n2) imbalance strategies x models (5-fold x 3 seeds)")
    res = strategy_experiments(X_train, y_train)
    res.to_csv(OUT_DIR / "day04_strategy_results.csv", index=False)
    choice = choose(res)
    (OUT_DIR / "day04_choice.json").write_text(json.dumps(choice, indent=2))
    print(f"\n   chosen: {choice['model']} / {choice['strategy']}  (tied: {choice['tied_candidates']})")

    print("\n3) segment robustness for the chosen configuration")
    seg = segment_experiments(X_train, y_train, choice)
    seg.to_csv(OUT_DIR / "day04_segment_results.csv", index=False)
    show = seg[["scope", "segment", "n", "positives", "pr_auc_baseline", "pr_auc_engineered", "delta_pr_auc", "low_support"]]
    print(show.round(3).to_string(index=False))

    # ---- plots ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, metric, title in zip(axes, ("pr_auc_mean", "recall_mean", "f1_mean"),
                                 ("PR-AUC (threshold-free)", "Recall @ 0.5", "F1 @ 0.5")):
        pivot = res.pivot(index="strategy", columns="model", values=metric).loc[list(STRATEGIES)]
        pivot.plot.bar(ax=ax, rot=30)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("Day 04 - imbalance strategies (5-fold x 3 seeds, all engineered features)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "day04_strategies.png", dpi=130)
    plt.close(fig)

    plot = seg.assign(label=seg["scope"] + ": " + seg["segment"])
    plot = plot[plot["scope"] != "overall"].sort_values("label")
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(plot) + 1.5))
    y_pos = np.arange(len(plot))
    ax.barh(y_pos + 0.2, plot["pr_auc_engineered"], height=0.4, label="engineered", color="#3b7dd8")
    ax.barh(y_pos - 0.2, plot["pr_auc_baseline"], height=0.4, label="baseline", color="#bbbbbb")
    ax.errorbar(plot["pr_auc_engineered"], y_pos + 0.2, xerr=plot["pr_auc_seed_std"].fillna(0), fmt="none", ecolor="k", lw=0.8)
    ax.set_yticks(y_pos, plot["label"], fontsize=7)
    overall = seg[seg["scope"] == "overall"].iloc[0]
    ax.axvline(overall["pr_auc_engineered"], color="#3b7dd8", ls="--", lw=0.8)
    ax.set_xlabel("out-of-fold PR-AUC (mean over 3 seeds; bar = seed std)")
    ax.set_title("Day 04 - performance by segment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "day04_segments.png", dpi=130)
    plt.close(fig)
    print("\nDay 04 done.")


if __name__ == "__main__":
    main()
