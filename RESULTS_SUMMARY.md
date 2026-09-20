# Results summary (Days 1-5)

All experiment numbers except the hold-out table come from cross-validation on the 80% training split.

## Day 02 - pipeline sanity check
| model | pr_auc_mean | pr_auc_std | roc_auc_mean | roc_auc_std | f1_mean | f1_std |
|---|---|---|---|---|---|---|
| dummy (prevalence) | 0.105 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 |
| pipeline: base columns only | 0.157 | 0.031 | 0.602 | 0.055 | 0.000 | 0.000 |
| pipeline: all engineered features | 0.436 | 0.053 | 0.791 | 0.040 | 0.342 | 0.016 |

## Day 03 - baseline vs engineered features (5-fold x 2 repeats, paired folds)
`delta_pr_auc` is against the reference (baseline for add_one, full for drop_one, baseline+aggregate for add_on_aggregate).

| model | experiment | group | pr_auc_mean | pr_auc_std | roc_auc_mean | delta_pr_auc | delta_ci95_low | delta_ci95_high | folds_better_than_ref |
|---|---|---|---|---|---|---|---|---|---|
| logreg | baseline | - | 0.156 | 0.022 | 0.601 | 0.000 | 0.000 | 0.000 | 0.000 |
| logreg | full | all groups | 0.439 | 0.040 | 0.795 | 0.283 | 0.260 | 0.305 | 1.000 |
| logreg | add_one | datetime | 0.151 | 0.017 | 0.599 | -0.006 | -0.011 | -0.000 | 0.200 |
| logreg | drop_one | datetime | 0.445 | 0.040 | 0.798 | 0.006 | 0.002 | 0.010 | 0.900 |
| logreg | add_one | aggregate | 0.440 | 0.043 | 0.793 | 0.284 | 0.261 | 0.307 | 1.000 |
| logreg | drop_one | aggregate | 0.436 | 0.040 | 0.794 | -0.004 | -0.010 | 0.002 | 0.400 |
| logreg | add_one | ratio | 0.229 | 0.041 | 0.702 | 0.073 | 0.054 | 0.092 | 1.000 |
| logreg | drop_one | ratio | 0.441 | 0.042 | 0.796 | 0.001 | -0.001 | 0.004 | 0.400 |
| logreg | add_one | interaction | 0.420 | 0.041 | 0.776 | 0.263 | 0.239 | 0.287 | 1.000 |
| logreg | drop_one | interaction | 0.434 | 0.042 | 0.792 | -0.005 | -0.011 | 0.002 | 0.300 |
| logreg | add_one | categorical | 0.305 | 0.043 | 0.758 | 0.149 | 0.127 | 0.171 | 1.000 |
| logreg | drop_one | categorical | 0.438 | 0.039 | 0.794 | -0.002 | -0.005 | 0.001 | 0.400 |
| logreg | add_on_aggregate | datetime | 0.437 | 0.045 | 0.793 | -0.003 | -0.007 | 0.001 | 0.300 |
| logreg | add_on_aggregate | ratio | 0.438 | 0.042 | 0.793 | -0.002 | -0.008 | 0.004 | 0.400 |
| logreg | add_on_aggregate | interaction | 0.446 | 0.040 | 0.798 | 0.006 | 0.000 | 0.011 | 0.600 |
| logreg | add_on_aggregate | categorical | 0.441 | 0.041 | 0.793 | 0.001 | -0.003 | 0.004 | 0.600 |
| hgb | baseline | - | 0.146 | 0.015 | 0.581 | 0.000 | 0.000 | 0.000 | 0.000 |
| hgb | full | all groups | 0.406 | 0.034 | 0.776 | 0.260 | 0.236 | 0.283 | 1.000 |
| hgb | add_one | datetime | 0.142 | 0.019 | 0.570 | -0.004 | -0.012 | 0.003 | 0.200 |
| hgb | drop_one | datetime | 0.407 | 0.037 | 0.779 | 0.001 | -0.006 | 0.007 | 0.600 |
| hgb | add_one | aggregate | 0.402 | 0.037 | 0.775 | 0.256 | 0.229 | 0.283 | 1.000 |
| hgb | drop_one | aggregate | 0.393 | 0.025 | 0.763 | -0.013 | -0.024 | -0.001 | 0.200 |
| hgb | add_one | ratio | 0.196 | 0.023 | 0.670 | 0.050 | 0.033 | 0.067 | 1.000 |
| hgb | drop_one | ratio | 0.403 | 0.034 | 0.772 | -0.003 | -0.009 | 0.003 | 0.500 |
| hgb | add_one | interaction | 0.381 | 0.034 | 0.748 | 0.235 | 0.213 | 0.258 | 1.000 |
| hgb | drop_one | interaction | 0.404 | 0.041 | 0.776 | -0.002 | -0.009 | 0.005 | 0.300 |
| hgb | add_one | categorical | 0.269 | 0.020 | 0.734 | 0.123 | 0.106 | 0.140 | 1.000 |
| hgb | drop_one | categorical | 0.404 | 0.038 | 0.775 | -0.002 | -0.006 | 0.003 | 0.500 |
| hgb | add_on_aggregate | datetime | 0.401 | 0.034 | 0.772 | -0.002 | -0.008 | 0.005 | 0.500 |
| hgb | add_on_aggregate | ratio | 0.397 | 0.034 | 0.775 | -0.006 | -0.013 | 0.002 | 0.400 |
| hgb | add_on_aggregate | interaction | 0.399 | 0.035 | 0.774 | -0.004 | -0.012 | 0.004 | 0.300 |
| hgb | add_on_aggregate | categorical | 0.401 | 0.042 | 0.775 | -0.002 | -0.010 | 0.007 | 0.600 |

## Day 04 - imbalance
| scope | n | positives | prevalence |
|---|---|---|---|
| train | 4800 | 506 | 0.1054 |
| hold-out | 1200 | 127 | 0.1058 |

Strategies (5-fold x 3 seeds; resampling happens inside training folds only):

| model | strategy | pr_auc_mean | pr_auc_fold_std | pr_auc_seed_std | roc_auc_mean | f1_mean | recall_mean | precision_mean |
|---|---|---|---|---|---|---|---|---|
| logreg | none | 0.440 | 0.046 | 0.006 | 0.798 | 0.336 | 0.226 | 0.665 |
| logreg | class_weight | 0.431 | 0.048 | 0.003 | 0.793 | 0.380 | 0.688 | 0.262 |
| logreg | oversample | 0.429 | 0.046 | 0.003 | 0.791 | 0.373 | 0.678 | 0.257 |
| logreg | undersample | 0.411 | 0.048 | 0.010 | 0.783 | 0.352 | 0.688 | 0.237 |
| logreg | smote | 0.421 | 0.054 | 0.007 | 0.788 | 0.365 | 0.668 | 0.252 |
| hgb | none | 0.410 | 0.053 | 0.005 | 0.777 | 0.339 | 0.232 | 0.640 |
| hgb | class_weight | 0.406 | 0.059 | 0.004 | 0.769 | 0.395 | 0.499 | 0.328 |
| hgb | oversample | 0.405 | 0.053 | 0.006 | 0.761 | 0.396 | 0.484 | 0.336 |
| hgb | undersample | 0.372 | 0.049 | 0.011 | 0.759 | 0.331 | 0.659 | 0.221 |
| hgb | smote | 0.418 | 0.051 | 0.006 | 0.778 | 0.363 | 0.265 | 0.588 |

Segments (out-of-fold, chosen configuration; `low_support` = fewer than 30 positives, treat with caution):

| scope | segment | n | positives | pr_auc_baseline | pr_auc_engineered | delta_pr_auc | pr_auc_seed_std | low_support |
|---|---|---|---|---|---|---|---|---|
| acquisition_channel | organic | 1457 | 137 | 0.113 | 0.384 | 0.270 | 0.005 | False |
| acquisition_channel | paid_search | 1175 | 162 | 0.192 | 0.459 | 0.267 | 0.009 | False |
| acquisition_channel | partner | 497 | 34 | 0.131 | 0.362 | 0.230 | 0.008 | False |
| acquisition_channel | referral | 705 | 69 | 0.147 | 0.401 | 0.254 | 0.022 | False |
| acquisition_channel | social | 966 | 104 | 0.132 | 0.510 | 0.378 | 0.005 | False |
| country | AE | 505 | 58 | 0.163 | 0.450 | 0.287 | 0.007 | False |
| country | CA | 318 | 25 | 0.137 | 0.454 | 0.317 | 0.007 | True |
| country | DE | 391 | 29 | 0.114 | 0.363 | 0.249 | 0.011 | True |
| country | IN | 466 | 47 | 0.151 | 0.508 | 0.357 | 0.006 | False |
| country | NZ | 89 | 11 | 0.203 | 0.503 | 0.300 | 0.019 | True |
| country | PK | 1452 | 177 | 0.155 | 0.422 | 0.267 | 0.013 | False |
| country | SE | 158 | 22 | 0.205 | 0.540 | 0.334 | 0.023 | True |
| country | UK | 479 | 47 | 0.136 | 0.433 | 0.297 | 0.007 | False |
| country | US | 942 | 90 | 0.182 | 0.396 | 0.214 | 0.007 | False |
| device | android | 2038 | 218 | 0.166 | 0.383 | 0.217 | 0.010 | False |
| device | ios | 1610 | 175 | 0.165 | 0.488 | 0.322 | 0.012 | False |
| device | missing | 233 | 17 | 0.084 | 0.360 | 0.276 | 0.031 | True |
| device | web | 919 | 96 | 0.114 | 0.462 | 0.348 | 0.003 | False |
| overall | all | 4800 | 506 | 0.149 | 0.430 | 0.281 | 0.007 | False |
| plan | basic | 2397 | 311 | 0.169 | 0.397 | 0.229 | 0.010 | False |
| plan | premium | 690 | 66 | 0.120 | 0.537 | 0.417 | 0.006 | False |
| plan | standard | 1713 | 129 | 0.099 | 0.454 | 0.355 | 0.005 | False |
| tenure_bucket | long (365d+) | 3580 | 361 | 0.138 | 0.423 | 0.286 | 0.010 | False |
| tenure_bucket | mid (90-365d) | 992 | 115 | 0.198 | 0.468 | 0.270 | 0.003 | False |
| tenure_bucket | new (<90d) | 228 | 30 | 0.166 | 0.401 | 0.235 | 0.014 | False |

## Day 05 - hold-out test set (used once)
| pipeline | threshold | pr_auc | roc_auc | f1 | recall | precision | balanced_acc | pr_auc_ci95 | roc_auc_ci95 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.5 | 0.137 | 0.568 | 0.000 | 0.000 | 0.000 | 0.500 | 0.109-0.183 | 0.512-0.615 |
| baseline | tuned (0.24) | 0.137 | 0.568 | 0.014 | 0.008 | 0.083 | 0.499 | 0.109-0.183 | 0.512-0.615 |
| full_engineered | 0.5 | 0.374 | 0.740 | 0.280 | 0.181 | 0.622 | 0.584 | 0.295-0.452 | 0.692-0.781 |
| full_engineered | tuned (0.24) | 0.374 | 0.740 | 0.358 | 0.378 | 0.340 | 0.646 | 0.295-0.452 | 0.692-0.781 |
| final | 0.5 | 0.368 | 0.734 | 0.307 | 0.197 | 0.694 | 0.593 | 0.291-0.446 | 0.684-0.777 |
| final | tuned (0.24) | 0.368 | 0.734 | 0.354 | 0.362 | 0.346 | 0.641 | 0.291-0.446 | 0.684-0.777 |

Paired bootstrap, final minus baseline: PR-AUC +0.147 to +0.309; ROC-AUC +0.107 to +0.232 (95% CI).
