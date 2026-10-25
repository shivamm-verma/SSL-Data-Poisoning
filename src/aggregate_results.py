"""
Once every friend has run their assigned configs and sent back their
*_result.json files (dropped into one shared results/ folder — see
the workflow notes), run this once to build a single master CSV
covering the whole (dataset x model x attack) grid.

Usage:
    python src/aggregate_results.py --results-dir ./results --out grid_summary.csv
"""
import argparse
import glob
import json
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--out", default="grid_summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*_result.json"))):
        with open(path) as f:
            r = json.load(f)
        row = {
            "run_id": r.get("run_id"), "dataset": r.get("dataset"), "model": r.get("model"),
            "attack": r.get("attack"), "rate": r.get("rate"),
            "clean_test_acc": r.get("best_clean_test_acc"), "asr": r.get("final_asr"),
            "epochs_run": r.get("epochs_run"),
        }
        detectors = r.get("detectors") or {}
        for det_name, metrics in detectors.items():
            row[f"{det_name}_auroc"] = metrics.get("auroc")
            row[f"{det_name}_precision_at_k"] = metrics.get("precision_at_k")
        rows.append(row)

    if not rows:
        print(f"[aggregate] no *_result.json files found in {args.results_dir}")
        return

    df = pd.DataFrame(rows).sort_values(["dataset", "model", "attack", "rate"])
    df.to_csv(args.out, index=False)
    print(f"[aggregate] {len(df)} runs -> {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
