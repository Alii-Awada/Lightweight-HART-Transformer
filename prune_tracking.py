import argparse
import datetime as dt
import glob
import os
import re

try:
    import hickle as hkl
    _HAS_HICKLE = True
except ImportError:
    _HAS_HICKLE = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import json

TRACKING_DIR = "tracking"
TRACKING_CSV = "prune_tracking.csv"
TRACKING_PNG = "prune_tracking_graph.png"
TRACKING_COMPARE_PNG = "prune_before_after_comparison.png"

def tracking_columns():
    return [
        "timestamp",
        "stage",
        "run_dir",
        "train_acc",
        "val_acc_peak",  # renamed from val_acc (was peak-epoch, not EMA eval)
        "weighted_f1",
        "micro_f1",
        "macro_f1",
        "size_mb",
        "flops_operations",
        "params",
        "compression_ratio",
        "complexity_matrix",
        # Test-split metadata (populated when run_dir is a main.py output)
        "test_fold_index",
        "test_n_folds",
        "test_split_pct",
    ]

def find_run_dirs_by_mtime(root_dir):
    """Return all dirs containing GlobalACC.csv sorted newest first."""
    candidates = glob.glob(os.path.join(root_dir, "**", "GlobalACC.csv"),
                           recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No GlobalACC.csv found under: {root_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return [os.path.dirname(p) for p in candidates]


def latest_run_dir(root_dir):
    return find_run_dirs_by_mtime(root_dir)[0]


def second_latest_run_dir(root_dir):
    dirs = find_run_dirs_by_mtime(root_dir)
    if len(dirs) < 2:
        raise FileNotFoundError(
            "Only one run found — cannot auto-detect a 'before' run. "
            "Pass --before-run-dir explicitly."
        )
    return dirs[1]

def parse_history_metrics(run_dir):
    history_path = os.path.join(run_dir, "history.hkl")
    if not os.path.exists(history_path):
        return None, None
    if not _HAS_HICKLE:
        print("[Warning] hickle is not installed — cannot read history.hkl. "
              "Install with: pip install hickle")
        return None, None
    history = hkl.load(history_path)
    train_acc = max(history.get("accuracy", [0.0])) * 100.0
    val_acc = max(history.get("val_accuracy", [0.0])) * 100.0
    # NOTE: val_acc_peak is the highest validation accuracy across all epochs,
    # which may differ from the EMA-checkpoint accuracy used for final eval.
    return round(train_acc, 4), round(val_acc, 4)


def _safe_float(v):
    """Convert a value to float, returning None on failure."""
    if v is None:
        return None
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None


def parse_globalacc_metrics(run_dir):
    defaults = {
        "weighted_f1": None, "micro_f1": None, "macro_f1": None,
        "size_mb": None, "flops_operations": None,
        "params": None, "complexity_matrix": None,
        "test_fold_index": None, "test_n_folds": None, "test_split_pct": None,
    }
    global_path = os.path.join(run_dir, "GlobalACC.csv")
    if not os.path.exists(global_path):
        return defaults

    with open(global_path, "r", encoding="utf-8") as fh:
        df = pd.read_csv(fh, header=None, names=["key", "value"])

    # Strip leading/trailing whitespace and the leading \n from keys
    df["key"] = df["key"].astype(str).str.strip().str.lstrip("\n").str.strip()
    lookup = dict(zip(df["key"], df["value"]))

    return {
        "weighted_f1": _safe_float(lookup.get("Test weighted f1:")),
        "micro_f1": _safe_float(lookup.get("Test micro f1:")),
        "macro_f1": _safe_float(lookup.get("Test macro f1:")),
        # Both key variants written by main.py — take whichever is present
        "size_mb": _safe_float(
            lookup.get("Size (MB):") or
            lookup.get("Model size (MB):")),
        "flops_operations": _safe_float(
            lookup.get("FLOPs operations (batch=1):") or
            lookup.get("FLOPs (batch=1):")),
        "params": _safe_float(lookup.get("Model params:")),
        "complexity_matrix": str(lookup.get("Complexity matrix:", "")).strip() or None,
        # Test-split metadata written by main.py (utils.py fold info)
        "test_fold_index": _safe_float(lookup.get("Test split (fold index):")),
        "test_n_folds":    _safe_float(lookup.get("Test split (total folds):")),
        "test_split_pct":  _safe_float(lookup.get("Test split size (%):")),
    }


def parse_complexity_matrix(run_dir):
    matrix_path = os.path.join(run_dir, "ComplexityMatrix.csv")
    if not os.path.exists(matrix_path):
        return {}
    parsed = {}
    matrix_df = pd.read_csv(matrix_path)
    for _, row in matrix_df.iterrows():
        metric = str(row.get("metric", "")).strip().lower()
        value = pd.to_numeric(str(row.get("value", "")).strip(), errors="coerce")
        if metric == "size_mb":
            parsed["size_mb"] = value
        elif metric == "flops_batch1":
            parsed["flops_operations"] = value
        elif metric == "params":
            parsed["params"] = value
    return parsed


def parse_compression_summary(run_dir):
    """
    Read compression_summary.json written by the fixed prune_distill.py.
    Returns dict or {} if not present.
    """
    summary_path = os.path.join(run_dir, "compression_summary.json")
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_prune_distill_csv(run_dir):
    """
    Read the best compressed-cycle metrics from iterative_prune_distill.csv
    (produced by prune_distill.py).  Returns a partial metrics dict (or {}).

    The best cycle is chosen by highest val_acc among stage == 'compressed' rows.
    Fields returned mirror what parse_globalacc_metrics returns so collect_metrics
    can merge them with the same priority logic.
    """
    csv_path = os.path.join(run_dir, "iterative_prune_distill.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return {}
    if "stage" not in df.columns:
        return {}
    compressed = df[df["stage"] == "compressed"].copy()
    if compressed.empty:
        return {}
    for col in ("val_acc", "weighted_f1", "micro_f1", "macro_f1", "size_mb", "params"):
        if col in compressed.columns:
            compressed[col] = pd.to_numeric(compressed[col], errors="coerce")
    best_row = compressed.loc[compressed["val_acc"].idxmax()]
    return {
        "weighted_f1": _safe_float(best_row.get("weighted_f1")),
        "micro_f1":    _safe_float(best_row.get("micro_f1")),
        "macro_f1":    _safe_float(best_row.get("macro_f1")),
        "size_mb":     _safe_float(best_row.get("size_mb")),
        "params":      _safe_float(best_row.get("params")),
        # expose best val_acc as val_acc_peak so tracking uses the same column
        "val_acc_peak": _safe_float(best_row.get("val_acc")),
        # flops not recorded by prune_distill.py
        "flops_operations": None,
        "complexity_matrix": None,
    }


def collect_metrics(run_dir):
    # --- Source 1: main.py run artifacts ---
    train_acc, val_acc_peak_hist = parse_history_metrics(run_dir)
    global_metrics = parse_globalacc_metrics(run_dir)
    matrix_metrics = parse_complexity_matrix(run_dir)
    summary = parse_compression_summary(run_dir)

    # --- Source 2: prune_distill.py run artifact (iterative_prune_distill.csv) ---
    # Takes priority over GlobalACC / ComplexityMatrix when present, because
    # prune_distill.py output dirs don't contain those main.py files.
    prune_metrics = parse_prune_distill_csv(run_dir)

    def first_not_none(*values):
        for v in values:
            if v is not None:
                return v
        return None

    weighted_f1 = first_not_none(prune_metrics.get("weighted_f1"), global_metrics["weighted_f1"])
    micro_f1    = first_not_none(prune_metrics.get("micro_f1"),    global_metrics["micro_f1"])
    macro_f1    = first_not_none(prune_metrics.get("macro_f1"),    global_metrics["macro_f1"])
    val_acc_peak = first_not_none(prune_metrics.get("val_acc_peak"), val_acc_peak_hist)

    size_mb = first_not_none(
        prune_metrics.get("size_mb"),
        global_metrics["size_mb"],
        matrix_metrics.get("size_mb"),
    )
    flops = first_not_none(
        global_metrics["flops_operations"],
        matrix_metrics.get("flops_operations"),
    )
    params = first_not_none(
        prune_metrics.get("params"),
        global_metrics["params"],
        matrix_metrics.get("params"),
    )

    compression_ratio = summary.get("compression_ratio", None)

    return {
        "train_acc": train_acc,
        "val_acc_peak": val_acc_peak,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "size_mb": size_mb,
        "flops_operations": flops,
        "params": params,
        "compression_ratio": compression_ratio,
        "complexity_matrix": global_metrics["complexity_matrix"] or "ComplexityMatrix.csv",
        "test_fold_index": global_metrics.get("test_fold_index"),
        "test_n_folds":    global_metrics.get("test_n_folds"),
        "test_split_pct":  global_metrics.get("test_split_pct"),
    }

def ensure_tracking_file(path):
    columns = tracking_columns()
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    existing_df = pd.read_csv(path)
    missing_cols = [c for c in columns if c not in existing_df.columns]
    if not missing_cols:
        return
    for col in missing_cols:
        existing_df[col] = np.nan
    existing_df = existing_df[columns]
    existing_df.to_csv(path, index=False)


def append_record(path, record):
    """
    FIX W6: append a single row using csv append mode.
    No full-file read/write — safe for back-to-back calls in --stage both.
    """
    row_df = pd.DataFrame([record], columns=tracking_columns())
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    row_df.to_csv(path, mode="a", index=False, header=write_header)


def load_records(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=tracking_columns())
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return pd.DataFrame(columns=tracking_columns())
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    numeric_cols = [
        "train_acc", "val_acc_peak", "weighted_f1",
        "micro_f1", "macro_f1", "size_mb",
        "flops_operations", "params", "compression_ratio",
        # Fold metadata must also be numeric so pd.notna() and int() work correctly
        "test_fold_index", "test_n_folds", "test_split_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "complexity_matrix" not in df.columns:
        df["complexity_matrix"] = np.nan
    return df

def print_comparison_table(df):
    """Print a formatted before/after comparison to stdout."""
    if df.empty:
        return
    before_df = df[df["stage"] == "before"]
    after_df = df[df["stage"] == "after"]
    if before_df.empty or after_df.empty:
        return

    before = before_df.iloc[-1]
    after = after_df.iloc[-1]

    def fmt(v, decimals=2):
        if pd.isna(v) or v is None:
            return "N/A"
        return f"{float(v):.{decimals}f}"

    def delta(b, a, higher_is_better=True):
        if pd.isna(b) or pd.isna(a):
            return ""
        d = float(a) - float(b)
        sign = "+" if d >= 0 else ""
        arrow = ("▲" if d > 0 else "▼") if higher_is_better else \
            ("▼" if d > 0 else "▲")
        return f"  {arrow} {sign}{d:.2f}"

    w = 62
    print("\n" + "=" * w)
    print(" Before / After Compression Comparison")
    print("=" * w)
    print(f"{'Metric':<26} {'Before':>10} {'After':>10} {'Change':>12}")
    print("-" * w)
    rows = [
        ("Val acc peak (%)", before.get("val_acc_peak"), after.get("val_acc_peak"), True),
        ("Weighted F1 (%)", before.get("weighted_f1"), after.get("weighted_f1"), True),
        ("Macro F1 (%)", before.get("macro_f1"), after.get("macro_f1"), True),
        ("Size (MB)", before.get("size_mb"), after.get("size_mb"), False),
        ("Params", before.get("params"), after.get("params"), False),
        ("Compression ratio", None, after.get("compression_ratio"), True),
    ]
    for label, bval, aval, hib in rows:
        b_str = fmt(bval) if bval is not None else "N/A"
        a_str = fmt(aval) if aval is not None else "N/A"
        d_str = delta(bval, aval, hib) if bval is not None and aval is not None else ""
        print(f"  {label:<24} {b_str:>10} {a_str:>10} {d_str:>12}")

    # Show test-split provenance for the 'before' run when available
    fi = before.get("test_fold_index")
    nf = before.get("test_n_folds")
    sp = before.get("test_split_pct")
    if pd.notna(fi) and pd.notna(nf):
        pct_str = f"  ({sp}% of data)" if pd.notna(sp) else ""
        print(f"\n  Test split (before): fold {int(fi)} of {int(nf)}-fold CV{pct_str}")

    print("=" * w + "\n")

def plot_tracking(df, output_png, output_compare_png):
    if df.empty:
        return

    df = df.sort_values("timestamp").reset_index(drop=True)
    before_df = df[df["stage"] == "before"].copy()
    after_df = df[df["stage"] == "after"].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    if not before_df.empty and before_df["timestamp"].notna().any():
        ax1.plot(before_df["timestamp"], before_df["weighted_f1"],
                 marker="o", label="Before prune — weighted F1", color="#1F78B4")
        ax2.plot(before_df["timestamp"], before_df["val_acc_peak"],
                 marker="o", label="Before prune — val acc (peak)", color="#1F78B4")

    if not after_df.empty and after_df["timestamp"].notna().any():
        ax1.plot(after_df["timestamp"], after_df["weighted_f1"],
                 marker="s", linestyle="--", label="After prune — weighted F1",
                 color="#E31A1C")
        ax2.plot(after_df["timestamp"], after_df["val_acc_peak"],
                 marker="s", linestyle="--",
                 label="After prune — val acc (peak)", color="#E31A1C")

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
        ax.grid(alpha=0.3)
        ax.legend()

    ax1.set_ylabel("Weighted F1 (%)")
    ax1.set_title("Pruning Tracking — Weighted F1 over time")
    ax2.set_ylabel("Validation Accuracy — peak epoch (%)")
    ax2.set_title("Pruning Tracking — Validation Accuracy over time")
    ax2.set_xlabel("Timestamp")

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)

    latest_by_stage = (
        df.sort_values("timestamp")
        .groupby(["run_dir", "stage"], as_index=False)
        .tail(1)
    )
    # pivot_table instead of pivot — safe when the same run_dir appears more than
    # once for the same stage (aggfunc="last" keeps the most-recent value).
    compare_df = latest_by_stage.pivot_table(
        index="run_dir", columns="stage", values="weighted_f1", aggfunc="last"
    )
    compare_df.columns.name = None   # drop the "stage" axis label
    if compare_df.empty:
        return

    compare_df = compare_df.fillna(np.nan)
    x = np.arange(len(compare_df.index))
    width = 0.35
    before_vals = compare_df["before"].values \
        if "before" in compare_df.columns else np.full(len(x), np.nan)
    after_vals = compare_df["after"].values \
        if "after" in compare_df.columns else np.full(len(x), np.nan)

    fig2, ax = plt.subplots(figsize=(max(8, len(x) * 2), 6))
    bars_b = ax.bar(x - width / 2, before_vals, width,
                    label="Before prune", color="#1F78B4", alpha=0.85)
    bars_a = ax.bar(x + width / 2, after_vals, width,
                    label="After prune", color="#E31A1C", alpha=0.85)

    # Add value labels on bars
    for bar in bars_b:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_a:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [os.path.basename(r) for r in compare_df.index],
        rotation=20, ha="right"
    )
    ax.set_ylabel("Weighted F1 (%)")
    ax.set_title("Before vs After Pruning — Weighted F1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(output_compare_png, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig2)

def add_stage_record(stage, run_dir, tracking_csv):
    metrics = collect_metrics(run_dir)

    # Guard: refuse to write a record when every key metric is missing.
    # This happens when main.py has not been run yet (no GlobalACC.csv /
    # history.hkl in the directory), which would silently pollute the CSV.
    _key_metrics = ["val_acc_peak", "weighted_f1", "macro_f1", "size_mb", "params"]
    if all(metrics.get(k) is None for k in _key_metrics):
        print(
            f"[Warning] No metrics found in '{run_dir}'.\n"
            f"  Expected files: GlobalACC.csv, history.hkl, iterative_prune_distill.csv\n"
            f"  Has main.py / prune_distill.py been run for this directory yet?\n"
            f"  Skipping record — tracking CSV not updated."
        )
        return

    append_record(
        tracking_csv,
        {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "run_dir": os.path.abspath(run_dir),
            **metrics,
        },
    )

def main():
    parser = argparse.ArgumentParser(
        description="Unified pruning tracker: record before/after metrics and compare."
    )
    parser.add_argument(
        "--results-root", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "HART_Results"),
        help="Root directory containing training result folders.",
    )
    parser.add_argument(
        "--stage", type=str,
        choices=["before", "after", "both"],
        default="both",
        help=(
            "Stage to record.  "
            "'before'/'after': auto-detect latest run.  "
            "'both': requires --before-run-dir AND --after-run-dir, or auto-detects "
            "second-latest as before and latest as after."
        ),
    )
    parser.add_argument("--run-dir", type=str, default="",
                        help="Run dir for single-stage mode.")
    parser.add_argument("--before-run-dir", type=str, default="",
                        help="Run dir for the 'before' record (--stage both).")
    parser.add_argument("--after-run-dir", type=str, default="",
                        help="Run dir for the 'after' record (--stage both).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate plots from existing CSV only.")
    args = parser.parse_args()

    tracking_dir = os.path.join(args.results_root, TRACKING_DIR)
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_csv = os.path.join(tracking_dir, TRACKING_CSV)
    tracking_png = os.path.join(tracking_dir, TRACKING_PNG)
    tracking_compare_png = os.path.join(tracking_dir, TRACKING_COMPARE_PNG)
    ensure_tracking_file(tracking_csv)

    if not args.plot_only:
        if args.stage == "before":
            run_dir = args.run_dir if args.run_dir else latest_run_dir(args.results_root)
            add_stage_record("before", run_dir, tracking_csv)
            print(f"Added BEFORE record from: {run_dir}")

        elif args.stage == "after":
            run_dir = args.run_dir if args.run_dir else latest_run_dir(args.results_root)
            add_stage_record("after", run_dir, tracking_csv)
            print(f"Added AFTER record from: {run_dir}")

        else:  # both
            # FIX B5: determine before/after dirs explicitly
            if args.before_run_dir and args.after_run_dir:
                before_run = args.before_run_dir
                after_run = args.after_run_dir
            elif args.before_run_dir or args.after_run_dir:
                parser.error(
                    "--stage both: provide BOTH --before-run-dir and --after-run-dir, "
                    "or neither (auto-detect second-latest vs latest)."
                )
            else:
                # Auto-detect: second-latest = before, latest = after
                before_run = second_latest_run_dir(args.results_root)
                after_run = latest_run_dir(args.results_root)
                if before_run == after_run:
                    parser.error(
                        "Auto-detect found only one unique run directory. "
                        "Pass --before-run-dir and --after-run-dir explicitly."
                    )

            add_stage_record("before", before_run, tracking_csv)
            add_stage_record("after", after_run, tracking_csv)
            print(f"Added BEFORE record from: {before_run}")
            print(f"Added AFTER  record from: {after_run}")

    records = load_records(tracking_csv)
    plot_tracking(records, tracking_png, tracking_compare_png)
    print_comparison_table(records)

    print(f"Tracking CSV     : {tracking_csv}")
    print(f"Time-series plot : {tracking_png}")
    print(f"Comparison plot  : {tracking_compare_png}")


if __name__ == "__main__":
    main()
