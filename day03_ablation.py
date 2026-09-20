"""DAY 03 - Compare baseline features with engineered features (ablation study).

Experiments (5-fold x 2 repeats, identical folds for every experiment -> paired comparison):
    baseline      base columns only
    add-one       baseline + ONE feature group          -> how much does each family add on its own?
    full          baseline + ALL groups
    drop-one      full minus ONE feature group          -> how much is lost if a family is removed?
    add-on-agg    baseline + aggregate + ONE more group -> does anything add value on top of the
                                                           strongest family?  (groups overlap in
                                                           information, so drop-one alone can hide value)

Metric focus: PR-AUC (rare positive class), plus ROC-AUC and F1@0.5.  Deltas are computed fold-by-fold
against the reference (baseline for add-one, full for drop-one, baseline+aggregate for add-on-agg).

Output: outputs/day03_ablation_results.csv, outputs/day03_ablation.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ABLATION_REPEATS, OUT_DIR
from .data import load_splits
from .evaluation import cv_evaluate
from .features import ALL_GROUPS
from .pipeline import build_pipeline

CV_SEED = 0


def _run(model, groups, X, y):
    folds, _ = cv_evaluate(build_pipeline(groups=groups, model=model), X, y,
                           n_repeats=ABLATION_REPEATS, random_state=CV_SEED)
    return folds


def _row(model, kind, group, folds, ref):
    d = folds["pr_auc"].to_numpy() - ref["pr_auc"].to_numpy()
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return {"model": model, "experiment": kind, "group": group,
            "pr_auc_mean": folds["pr_auc"].mean(), "pr_auc_std": folds["pr_auc"].std(ddof=1),
            "roc_auc_mean": folds["roc_auc"].mean(), "roc_auc_std": folds["roc_auc"].std(ddof=1),
            "f1_mean": folds["f1"].mean(),
            "delta_pr_auc": d.mean(), "delta_ci95_low": d.mean() - half, "delta_ci95_high": d.mean() + half,
            "folds_better_than_ref": float((d > 0).mean()),
            "delta_roc_auc": (folds["roc_auc"].to_numpy() - ref["roc_auc"].to_numpy()).mean()}


def main():
    X_train, _, y_train, _ = load_splits()
    rows = []
    for model in ("logreg", "hgb"):
        print(f"\n=== model: {model} ===")
        baseline = _run(model, (), X_train, y_train)
        full = _run(model, ALL_GROUPS, X_train, y_train)
        rows.append(_row(model, "baseline", "-", baseline, baseline))
        rows.append(_row(model, "full", "all groups", full, baseline))
        agg_only = None
        for g in ALL_GROUPS:
            folds_g = _run(model, (g,), X_train, y_train)
            if g == "aggregate":
                agg_only = folds_g
            rows.append(_row(model, "add_one", g, folds_g, baseline))
            rest = tuple(x for x in ALL_GROUPS if x != g)
            rows.append(_row(model, "drop_one", g, _run(model, rest, X_train, y_train), full))
        for g in ALL_GROUPS:
            if g != "aggregate":
                rows.append(_row(model, "add_on_aggregate", g,
                                 _run(model, ("aggregate", g), X_train, y_train), agg_only))
        part = pd.DataFrame([r for r in rows if r["model"] == model])
        print(part[["experiment", "group", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "delta_pr_auc",
                    "folds_better_than_ref"]].round(3).to_string(index=False))

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "day03_ablation_results.csv", index=False)

    # ---- plot ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
    for i, model in enumerate(("logreg", "hgb")):
        sub = res[res["model"] == model]
        add = sub[sub["experiment"] == "add_one"]
        drop = sub[sub["experiment"] == "drop_one"]
        on_agg = sub[sub["experiment"] == "add_on_aggregate"]
        for ax, data, title in ((axes[i, 0], add, "add ONE group to baseline"),
                                (axes[i, 1], drop, "drop ONE group from full"),
                                (axes[i, 2], on_agg, "add ONE group on top of baseline+aggregate")):
            err = [data["delta_pr_auc"] - data["delta_ci95_low"], data["delta_ci95_high"] - data["delta_pr_auc"]]
            ax.barh(data["group"], data["delta_pr_auc"], xerr=err, color="#3b7dd8", capsize=3)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{model}: {title}", fontsize=9)
            ax.set_xlabel("delta PR-AUC (mean, 95% CI over folds)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "day03_ablation.png", dpi=130)
    plt.close(fig)
    print("\nDay 03 done -> outputs/day03_ablation_results.csv, day03_ablation.png")


if __name__ == "__main__":
    main()
