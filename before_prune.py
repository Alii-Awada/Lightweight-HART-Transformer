#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import datetime as dt
import glob
import os
import re

import hickle as hkl
import matplotlib.pyplot as plt


TRACKING_DIR = "tracking"
TRACKING_CSV = "prune_tracking.csv"
TRACKING_PNG = "prune_tracking_graph.png"


def latest_run_dir(root_dir):
    candidates = glob.glob(os.path.join(root_dir, "**", "GlobalACC.csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No GlobalACC.csv found under: {root_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.dirname(candidates[0])


def parse_history_metrics(run_dir):
    history_path = os.path.join(run_dir, "history.hkl")
    if not os.path.exists(history_path):
        return None, None
    history = hkl.load(history_path)
    train_acc = max(history.get("accuracy", [0.0])) * 100.0
    val_acc = max(history.get("val_accuracy", [0.0])) * 100.0
    return train_acc, val_acc


def parse_globalacc_metrics(run_dir):
    metrics = {"weighted_f1": None, "micro_f1": None, "macro_f1": None}
    global_path = os.path.join(run_dir, "GlobalACC.csv")
    if not os.path.exists(global_path):
        return metrics

    text = open(global_path, "r", encoding="utf-8").read()

    patterns = {
        "weighted_f1": r"Test weighted f1:\",\s*([0-9]+(?:\.[0-9]+)?)",
        "micro_f1": r"Test micro f1:\",\s*([0-9]+(?:\.[0-9]+)?)",
        "macro_f1": r"Test macro f1:\",\s*([0-9]+(?:\.[0-9]+)?)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def ensure_tracking_file(path):
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "stage",
                "run_dir",
                "train_acc",
                "val_acc",
                "weighted_f1",
                "micro_f1",
                "macro_f1",
            ]
        )


def append_record(path, record):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(record)


def to_float(v):
    try:
        if v in ("", None, "None"):
            return None
        return float(v)
    except Exception:
        return None


def load_records(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["train_acc"] = to_float(row["train_acc"])
            row["val_acc"] = to_float(row["val_acc"])
            row["weighted_f1"] = to_float(row["weighted_f1"])
            row["micro_f1"] = to_float(row["micro_f1"])
            row["macro_f1"] = to_float(row["macro_f1"])
            rows.append(row)
    return rows


def plot_tracking(records, output_png):
    before_rows = [r for r in records if r["stage"] == "before"]
    after_rows = [r for r in records if r["stage"] == "after"]

    plt.figure(figsize=(11, 7))

    ax1 = plt.subplot(2, 1, 1)
    if before_rows:
        ax1.plot(
            range(1, len(before_rows) + 1),
            [r["weighted_f1"] for r in before_rows],
            marker="o",
            label="Before prune - weighted F1",
        )
    if after_rows:
        ax1.plot(
            range(1, len(after_rows) + 1),
            [r["weighted_f1"] for r in after_rows],
            marker="o",
            label="After prune - weighted F1",
        )
    ax1.set_ylabel("Weighted F1 (%)")
    ax1.set_title("Pruning Tracking - Weighted F1")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    if before_rows:
        ax2.plot(
            range(1, len(before_rows) + 1),
            [r["val_acc"] for r in before_rows],
            marker="o",
            label="Before prune - val acc",
        )
    if after_rows:
        ax2.plot(
            range(1, len(after_rows) + 1),
            [r["val_acc"] for r in after_rows],
            marker="o",
            label="After prune - val acc",
        )
    ax2.set_xlabel("Run index by stage")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Pruning Tracking - Validation Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Track metrics before pruning and draw trend graph.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "HART_Results"),
        help="Path containing training result folders.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Specific run directory containing GlobalACC.csv and history.hkl.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else latest_run_dir(args.results_root)
    train_acc, val_acc = parse_history_metrics(run_dir)
    f1_metrics = parse_globalacc_metrics(run_dir)

    tracking_dir = os.path.join(args.results_root, TRACKING_DIR)
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_csv = os.path.join(tracking_dir, TRACKING_CSV)
    tracking_png = os.path.join(tracking_dir, TRACKING_PNG)

    ensure_tracking_file(tracking_csv)
    append_record(
        tracking_csv,
        [
            dt.datetime.now().isoformat(timespec="seconds"),
            "before",
            os.path.abspath(run_dir),
            train_acc,
            val_acc,
            f1_metrics["weighted_f1"],
            f1_metrics["micro_f1"],
            f1_metrics["macro_f1"],
        ],
    )

    records = load_records(tracking_csv)
    plot_tracking(records, tracking_png)
    print(f"Added BEFORE-prune record from: {run_dir}")
    print(f"Tracking CSV: {tracking_csv}")
    print(f"Tracking graph: {tracking_png}")


if __name__ == "__main__":
    main()
