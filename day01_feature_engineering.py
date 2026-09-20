"""DAY 01 - Build the engineered features and decide which ones are worth keeping.

Output: the feature-engineering module (src/features.py) + this report
    outputs/day01_feature_summary.csv       one row per feature (group, type, missingness, signal, filters)
    outputs/day01_correlation_heatmap.png   correlations between numeric engineered features
    outputs/day01_engineered_preview.csv    first rows of the engineered training matrix

All analysis below is done on the TRAINING split only.  The label is used just to *describe* signal
(mutual information); no fitted step of the pipeline sees y.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from .config import OUT_DIR, RANDOM_STATE
from .data import load_splits
from .features import CATEGORICAL_FEATURES, FEATURE_CATALOG, FeatureEngineer
from .pipeline import build_pipeline


def main():
    X_train, X_test, y_train, y_test = load_splits()
    print(f"train rows: {len(X_train)}, churn rate: {y_train.mean():.3f}")

    fe = FeatureEngineer().fit(X_train)
    F = fe.transform(X_train)
    print(f"engineered feature matrix: {F.shape[1]} columns")

    # ---- signal (mutual information, analysis only) --------------------------------------
    num_cols = [c for c in F.columns if c not in CATEGORICAL_FEATURES]
    cat_cols = [c for c in F.columns if c in CATEGORICAL_FEATURES]
    filled = F[num_cols].fillna(F[num_cols].median())
    cat_codes = pd.DataFrame(OrdinalEncoder().fit_transform(F[cat_cols].fillna("missing")), columns=cat_cols, index=F.index)
    M = pd.concat([filled, cat_codes], axis=1)
    mi = mutual_info_classif(M, y_train, discrete_features=[c in CATEGORICAL_FEATURES for c in M.columns],
                             random_state=RANDOM_STATE)
    mi = pd.Series(mi, index=M.columns, name="mutual_info")

    # ---- redundancy: fit the pipeline and see which columns the correlation filter removed ----
    pipe = build_pipeline().fit(X_train, y_train)
    dropped = set(pipe.named_steps["correlation_filter"].dropped_)

    def fully_dropped(feature):
        owned = [c for c in dropped if c == feature or c.startswith(feature + "_")]
        kept = [c for c in pipe.named_steps["correlation_filter"].keep_ if c == feature or c.startswith(feature + "_")]
        return bool(owned) and not kept

    rows = []
    for feat in F.columns:
        col = F[feat]
        top_share = col.value_counts(normalize=True, dropna=False).iloc[0]
        rows.append({
            "feature": feat,
            "missing_rate": col.isna().mean(),
            "n_unique": col.nunique(),
            "top_value_share": top_share,
            "mutual_info": mi[feat],
            "near_constant": top_share > 0.99,
            "redundant_corr_gt_0.95": fully_dropped(feat),
        })
    summary = FEATURE_CATALOG.merge(pd.DataFrame(rows), on="feature")
    summary["passes_filters"] = ~(summary["near_constant"] | summary["redundant_corr_gt_0.95"])
    summary = summary.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "day01_feature_summary.csv", index=False)
    F.head(200).to_csv(OUT_DIR / "day01_engineered_preview.csv")

    # ---- heatmap --------------------------------------------------------------------------
    corr = filled.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Day 01 - correlation between numeric engineered features (train)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "day01_correlation_heatmap.png", dpi=130)
    plt.close(fig)

    # ---- console report -------------------------------------------------------------------
    print("\nFeatures per group:")
    print(summary.groupby("group").agg(n=("feature", "size"), kept=("passes_filters", "sum"),
                                       mean_mi=("mutual_info", "mean")).round(4))
    print("\nTop 10 features by mutual information:")
    print(summary[["feature", "group", "mutual_info"]].head(10).round(4).to_string(index=False))
    print("\nRemoved by filters:", summary.loc[~summary["passes_filters"], "feature"].tolist())
    return summary


if __name__ == "__main__":
    main()
