"""Central configuration so every day's script uses the same settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"

RANDOM_STATE = 42
SNAPSHOT_DATE = "2024-06-30"   # "today" for the dataset: features use only data before this date
TARGET = "churn_30d"           # did the customer churn in the 30 days AFTER the snapshot?
ID_COL = "customer_id"
TEST_SIZE = 0.20               # final hold-out set, touched only on Day 05

# cross-validation used by the experiments
N_SPLITS = 5
ABLATION_REPEATS = 2
ROBUSTNESS_SEEDS = (11, 22, 33)

for _d in (OUT_DIR, DOCS_DIR, DATA_DIR):
    _d.mkdir(exist_ok=True)
