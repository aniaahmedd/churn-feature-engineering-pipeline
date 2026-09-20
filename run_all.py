"""Run the whole project:   python run_all.py            (data + Day 1..5 + tests)
                           python run_all.py --day 3    (a single day)
Days must be run in order the first time, because later days read earlier days' outputs."""
import argparse
import importlib
import subprocess
import sys

DAYS = {1: "src.day01_feature_engineering", 2: "src.day02_pipeline", 3: "src.day03_ablation",
        4: "src.day04_robustness", 5: "src.day05_review"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, choices=sorted(DAYS), help="run only this day")
    ap.add_argument("--no-tests", action="store_true", help="skip pytest at the end")
    args = ap.parse_args()

    from src.data import load_raw
    load_raw()                                    # creates data/*.csv on first run
    for day in ([args.day] if args.day else sorted(DAYS)):
        print(f"\n{'=' * 78}\nDAY {day:02d}  ({DAYS[day]})\n{'=' * 78}")
        importlib.import_module(DAYS[day]).main()
    if not args.day and not args.no_tests:
        print(f"\n{'=' * 78}\npytest\n{'=' * 78}")
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))


if __name__ == "__main__":
    main()
