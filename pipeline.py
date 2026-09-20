"""DAY 02 - Data transformation pipeline.

    FeatureEngineer  ->  ColumnTransformer(impute+scale numerics, impute+one-hot categoricals)
                     ->  CorrelationFilter  ->  [optional resampler]  ->  model

Everything that learns from data (medians, category frequencies, scaler mean/std, one-hot
categories, correlation filter, resampler) is a step in ONE sklearn/imblearn Pipeline, so:
  * fit() on a training fold learns from that fold only;
  * predict()/transform() on validation/test data re-uses those learned values;
  * cross-validation clones the pipeline for every fold (no state shared between folds).
"""
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .features import ALL_GROUPS, CATEGORICAL_FEATURES, CorrelationFilter, FeatureEngineer

MODELS = ("logreg", "hgb")
STRATEGIES = ("none", "class_weight", "oversample", "undersample", "smote")


# module-level selectors (picklable) - decide column type from the feature catalog
def _categorical_columns(df):
    return [c for c in df.columns if c in CATEGORICAL_FEATURES]


def _numeric_columns(df):
    return [c for c in df.columns if c not in CATEGORICAL_FEATURES]


def build_preprocessor(scale=True):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)),
    ])
    pre = ColumnTransformer(
        [("num", Pipeline(numeric_steps), _numeric_columns), ("cat", categorical, _categorical_columns)],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def build_model(name="logreg", class_weight=None, random_state=RANDOM_STATE):
    if name == "logreg":
        return LogisticRegression(C=0.5, max_iter=3000, class_weight=class_weight, random_state=random_state)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0,
                                              class_weight=class_weight, random_state=random_state)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def build_sampler(strategy, random_state=RANDOM_STATE):
    return {"oversample": RandomOverSampler(random_state=random_state),
            "undersample": RandomUnderSampler(random_state=random_state),
            "smote": SMOTE(random_state=random_state, k_neighbors=5)}.get(strategy)


def build_tail(model="logreg", strategy="none", corr_threshold=0.95, random_state=RANDOM_STATE):
    """Everything after feature engineering (used directly by Day 05 permutation importance)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    class_weight = "balanced" if strategy == "class_weight" else None
    steps = [("preprocess", build_preprocessor(scale=True)),
             ("correlation_filter", CorrelationFilter(corr_threshold))]
    sampler = build_sampler(strategy, random_state)
    if sampler is not None:
        steps.append(("resample", sampler))          # only ever applied while fitting
    steps.append(("model", build_model(model, class_weight, random_state)))
    return Pipeline(steps)


def build_pipeline(groups=ALL_GROUPS, model="logreg", strategy="none", drop_features=(),
                   corr_threshold=0.95, random_state=RANDOM_STATE):
    """Full leakage-safe pipeline: raw model frame in, churn probability out."""
    tail = build_tail(model, strategy, corr_threshold, random_state)
    features = FeatureEngineer(groups=tuple(groups), drop_features=tuple(drop_features))
    return Pipeline([("features", features)] + list(tail.steps))
