#!/usr/bin/env python
# coding: utf-8
"""
test_model.py — Evaluate any saved HART checkpoint or TFLite model on the
held-out test fold without rerunning training.

Usage examples
--------------
# Evaluate the best validation checkpoint (base model)
  python test_model.py --model base

# Evaluate the best training checkpoint
  python test_model.py --model base_train

# Evaluate the best pruned student
  python test_model.py --model student

# Evaluate a specific pruning cycle
  python test_model.py --model cycle --cycle 3

# Run the float32 TFLite model
  python test_model.py --model tflite

# Run the INT8 quantised TFLite model
  python test_model.py --model tflite_int8

# Evaluate a custom .weights.h5 path
  python test_model.py --model custom --ckpt path/to/weights.h5

# Override output dir or data dir
  python test_model.py --model student --output_dir HART_Results/pruning --data_dir ./
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

import model as hart_model
import utils

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_TEACHER_DIR = (
    "HART_Results/"
    "HART_16frameLength_16TimeStep_192ProjectionSize_0.0005LR_enhancedTok/"
    "PAMAP2"
)
DEFAULT_PRUNING_DIR = "HART_Results/pruning"
DEFAULT_TFLITE      = os.path.join(DEFAULT_TEACHER_DIR, "HART.tflite")
DEFAULT_TFLITE_INT8 = os.path.join(DEFAULT_TEACHER_DIR, "HART_int8.tflite")
METADATA_FILE       = os.path.join(DEFAULT_TEACHER_DIR, "run_metadata.json")
ACTIVITY_LABELS_TXT = "datasetStandardized/PAMAP2/activity_labels.txt"

# ── Activity labels (PAMAP2, 10 active classes across 18-dim output) ──────────
ACTIVITY_NAMES = {
    0: "lying",
    1: "sitting",
    2: "standing",
    3: "walking",
    4: "running",
    5: "cycling",
    6: "nordic_walking",
    7: "ascending_stairs",
    8: "descending_stairs",
    17: "rope_jumping",
}


def load_activity_names():
    """Read activity_labels.txt; fall back to ACTIVITY_NAMES dict."""
    names = dict(ACTIVITY_NAMES)
    if os.path.isfile(ACTIVITY_LABELS_TXT):
        with open(ACTIVITY_LABELS_TXT) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    names[int(parts[0])] = parts[2]
    return names


def load_metadata(path=METADATA_FILE):
    """Load run_metadata.json; return dict."""
    if not os.path.isfile(path):
        print(f"[Warning] run_metadata.json not found at {path} — using defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_compression_summary(pruning_dir=DEFAULT_PRUNING_DIR):
    path = os.path.join(pruning_dir, "compression_summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(seed=1, data_dir="./", class_count_override=None):
    """Load PAMAP2 fold-2 test split matching the training setup."""
    client_count = utils.returnClientByDataset("PAMAP2")
    dataset_loader = utils.loadDataset("PAMAP2", client_count, "BALANCED", seed, data_dir)

    train_x = dataset_loader.centralTrainData
    train_y = dataset_loader.centralTrainLabel
    test_x  = dataset_loader.centralTestData
    test_y  = dataset_loader.centralTestLabel

    train_x, dev_x, train_y, dev_y = train_test_split(
        train_x, train_y,
        test_size=0.125,
        random_state=seed,
        stratify=train_y,
    )

    detected = len(np.unique(np.hstack((train_y, dev_y, test_y))))
    class_count = class_count_override if class_count_override is not None else detected

    train_y_oh = tf.one_hot(train_y, class_count).numpy().astype(np.float32)
    dev_y_oh   = tf.one_hot(dev_y,   class_count).numpy().astype(np.float32)
    test_y_oh  = tf.one_hot(test_y,  class_count).numpy().astype(np.float32)

    return (
        train_x.astype(np.float32), train_y_oh,
        dev_x.astype(np.float32),   dev_y_oh,
        test_x.astype(np.float32),  test_y_oh,
        test_y,   # raw integer labels for sklearn metrics
        class_count,
    )


def build_hart(input_shape, class_count, meta, proj_dim=None, filter_heads=None):
    """Reconstruct HART from run_metadata.json fields (or override)."""
    proj_dim     = proj_dim     or meta.get("projection_dim",     192)
    filter_heads = filter_heads or meta.get("filter_heads",       4)
    frame_length = meta.get("frame_length", 16)
    time_step    = meta.get("time_step",    16)
    conv_kernels = meta.get("conv_kernels", [3, 7, 15, 31, 31, 31])
    if isinstance(conv_kernels, str):
        conv_kernels = [int(k) for k in conv_kernels.split(",")]

    return hart_model.HART(
        input_shape,
        class_count,
        projection_dim=proj_dim,
        patchSize=frame_length,
        timeStep=time_step,
        filterAttentionHead=filter_heads,
        convKernels=conv_kernels,
        dropout_rate=meta.get("dropout_rate",      0.15),
        attention_dropout=meta.get("attention_dropout", 0.08),
        mlp_dropout=meta.get("mlp_dropout",        0.1),
        token_dropout=meta.get("token_dropout",    0.05),
        drop_path_rate=meta.get("drop_path_rate",  0.12),
        useTokens=meta.get("token_based",          False),
        useEnhancedTokenizer=meta.get("enhanced_tokenizer", True),
    )


def compute_model_size_mb(m):
    total = sum(int(np.prod(w.shape)) * np.dtype(w.dtype).itemsize for w in m.weights)
    return round(total / 1024 / 1024, 4)


def print_results(label, ckpt_path, val_acc, test_acc,
                  y_true, y_pred, class_count, params, size_mb,
                  activity_names, elapsed_ms):
    """Pretty-print full evaluation results."""
    wf1  = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    mif1 = f1_score(y_true, y_pred, average="micro",    zero_division=0) * 100
    maf1 = f1_score(y_true, y_pred, average="macro",    zero_division=0) * 100

    print(f"\n{'=' * 64}")
    print(f"  {label}")
    print(f"{'=' * 64}")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Parameters   : {params:,}   |   Size: {size_mb:.2f} MB")
    print(f"  Inference time (test set, {y_true.shape[0]:,} samples): {elapsed_ms:.1f} ms")
    print(f"{'=' * 64}")
    print(f"  Val  accuracy : {val_acc  * 100:.2f}%")
    print(f"  Test accuracy : {test_acc * 100:.2f}%")
    print(f"  Weighted F1   : {wf1:.2f}%")
    print(f"  Micro    F1   : {mif1:.2f}%")
    print(f"  Macro    F1   : {maf1:.2f}%")
    print(f"  Test samples  : {y_true.shape[0]:,}")
    print(f"{'=' * 64}")

    # Per-class F1 with activity names
    unique = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [activity_names.get(i, f"class_{i}") for i in unique]
    report = classification_report(
        y_true, y_pred,
        labels=unique,
        target_names=target_names,
        zero_division=0,
    )
    print("\n  Per-class breakdown:")
    for line in report.splitlines():
        print("  " + line)

    # Confusion matrix (compact)
    cm = confusion_matrix(y_true, y_pred, labels=unique)
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    header = "  " + "".join(f"{str(c):>6}" for c in unique)
    print(header)
    for i, row in zip(unique, cm):
        name = activity_names.get(i, f"c{i}")[:12]
        print(f"  {name:<13}" + "".join(f"{v:>6}" for v in row))
    print()


# ── Keras model evaluation ────────────────────────────────────────────────────
def eval_keras(ckpt_path, input_shape, class_count, meta,
               dev_x, dev_y, test_x, test_y_oh, test_y_raw,
               activity_names, label,
               proj_dim=None, filter_heads=None):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    m = build_hart(input_shape, class_count, meta, proj_dim, filter_heads)
    m.build((None,) + input_shape)
    m.load_weights(ckpt_path)
    m.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    _, val_acc  = m.evaluate(dev_x,  dev_y,    verbose=0)
    _, test_acc = m.evaluate(test_x, test_y_oh, verbose=0)

    t0 = time.perf_counter()
    logits = m(test_x, training=False).numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    y_pred = np.argmax(logits, axis=-1)
    params  = m.count_params()
    size_mb = compute_model_size_mb(m)

    print_results(label, ckpt_path, val_acc, test_acc,
                  test_y_raw, y_pred, class_count, params, size_mb,
                  activity_names, elapsed_ms)


# ── TFLite evaluation ─────────────────────────────────────────────────────────
def eval_tflite(tflite_path, test_x, test_y_oh, test_y_raw,
                dev_x, dev_y_oh, activity_names, label):
    if not os.path.isfile(tflite_path):
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    size_mb = round(os.path.getsize(tflite_path) / 1024 / 1024, 4)

    def _infer_batch(data):
        preds = []
        for sample in data:
            x = sample[np.newaxis].astype(inp_detail["dtype"])
            if inp_detail["dtype"] == np.int8:
                scale, zp = inp_detail["quantization"]
                x = np.round(x / scale + zp).clip(-128, 127).astype(np.int8)
            interp.set_tensor(inp_detail["index"], x)
            interp.invoke()
            out = interp.get_tensor(out_detail["index"])[0]
            if out_detail["dtype"] == np.int8:
                scale, zp = out_detail["quantization"]
                out = (out.astype(np.float32) - zp) * scale
            preds.append(np.argmax(out))
        return np.array(preds)

    # Warmup (50 samples)
    _ = _infer_batch(test_x[:50])

    # Val accuracy (dev set)
    val_preds = _infer_batch(dev_x)
    val_true  = np.argmax(dev_y_oh, axis=-1)
    val_acc   = np.mean(val_preds == val_true)

    # Test set
    t0 = time.perf_counter()
    y_pred = _infer_batch(test_x)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    test_acc = np.mean(y_pred == test_y_raw)
    params = 0  # not accessible from TFLite

    print_results(label, tflite_path, val_acc, test_acc,
                  test_y_raw, y_pred, 0, params, size_mb,
                  activity_names, elapsed_ms)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved HART model on the test fold.")
    parser.add_argument("--model", type=str, default="base",
                        choices=["base", "base_train", "student", "cycle",
                                 "tflite", "tflite_int8", "custom"],
                        help=(
                            "Which model to evaluate:\n"
                            "  base       — bestValcheckpoint.weights.h5 (best val, base model)\n"
                            "  base_train — bestTrain.weights.h5 (best train, base model)\n"
                            "  student    — best_pruned_student.weights.h5\n"
                            "  cycle      — student_cycle_N.weights.h5 (set --cycle N)\n"
                            "  tflite     — float32 TFLite (HART.tflite)\n"
                            "  tflite_int8— INT8 quantised TFLite (HART_int8.tflite)\n"
                            "  custom     — arbitrary path, set --ckpt\n"
                        ))
    parser.add_argument("--cycle",      type=int, default=1,
                        help="Cycle number for --model cycle (default: 1)")
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Path to a custom .weights.h5 file (used with --model custom)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_PRUNING_DIR,
                        help="Directory containing pruning checkpoints")
    parser.add_argument("--teacher_dir",type=str, default=DEFAULT_TEACHER_DIR,
                        help="Directory containing base model checkpoint and metadata")
    parser.add_argument("--data_dir",   type=str, default="./",
                        help="Root directory containing datasetStandardized/")
    parser.add_argument("--seed",       type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load metadata & activity labels ──────────────────────────────────────
    meta_path = os.path.join(args.teacher_dir, "run_metadata.json")
    meta = load_metadata(meta_path)
    activity_names = load_activity_names()

    class_count = int(meta.get("activity_count", 18))
    input_shape_list = meta.get("input_shape", [128, 6])
    input_shape = tuple(input_shape_list)

    print(f"\n[Info] Dataset  : PAMAP2  |  Classes: {class_count}  |  Input: {input_shape}")
    print(f"[Info] Loading test fold (fold 2, 20%)...")

    # ── Load data ─────────────────────────────────────────────────────────────
    _, _, dev_x, dev_y, test_x, test_y_oh, test_y_raw, class_count = load_data(
        seed=args.seed,
        data_dir=args.data_dir,
        class_count_override=class_count,
    )
    print(f"[Info] Test  samples: {test_x.shape[0]:,}  |  Dev samples: {dev_x.shape[0]:,}")

    # ── Route to correct model ────────────────────────────────────────────────
    if args.model == "base":
        ckpt = os.path.join(args.teacher_dir, "bestValcheckpoint.weights.h5")
        eval_keras(ckpt, input_shape, class_count, meta,
                   dev_x, dev_y, test_x, test_y_oh, test_y_raw,
                   activity_names, label="Base HART — Best Validation Checkpoint")

    elif args.model == "base_train":
        ckpt = os.path.join(args.teacher_dir, "bestTrain.weights.h5")
        eval_keras(ckpt, input_shape, class_count, meta,
                   dev_x, dev_y, test_x, test_y_oh, test_y_raw,
                   activity_names, label="Base HART — Best Training Checkpoint")

    elif args.model == "student":
        summary = load_compression_summary(args.output_dir)
        proj_dim     = summary.get("best_projection_dim", 176)
        filter_heads = summary.get("best_filter_heads",   4)
        ckpt = os.path.join(args.output_dir, "best_pruned_student.weights.h5")
        eval_keras(ckpt, input_shape, class_count, meta,
                   dev_x, dev_y, test_x, test_y_oh, test_y_raw,
                   activity_names,
                   label=f"Pruned Student — best (proj={proj_dim}, heads={filter_heads})",
                   proj_dim=proj_dim, filter_heads=filter_heads)

    elif args.model == "cycle":
        summary = load_compression_summary(args.output_dir)
        # Compute cycle dimensions the same way prune_distill.py does
        prune_step      = 0.10
        head_prune_step = 0.10
        base_proj       = int(meta.get("projection_dim",   192))
        base_heads      = int(meta.get("filter_heads",     4))
        cycle = args.cycle
        proj_factor  = max(0.25, 1.0 - prune_step * cycle)
        head_factor  = max(0.25, 1.0 - head_prune_step * cycle)
        proj_dim     = max(64, int(round(base_proj  * proj_factor  / 16.0) * 16))
        filter_heads = max(1,  int(round(base_heads * head_factor)))
        while (proj_dim // 2) % filter_heads != 0 and filter_heads > 1:
            filter_heads -= 1
        ckpt = os.path.join(args.output_dir, f"student_cycle_{cycle}.weights.h5")
        eval_keras(ckpt, input_shape, class_count, meta,
                   dev_x, dev_y, test_x, test_y_oh, test_y_raw,
                   activity_names,
                   label=f"Pruned Student — cycle {cycle} (proj={proj_dim}, heads={filter_heads})",
                   proj_dim=proj_dim, filter_heads=filter_heads)

    elif args.model == "tflite":
        tflite_path = os.path.join(args.teacher_dir, "HART.tflite")
        eval_tflite(tflite_path, test_x, test_y_oh, test_y_raw,
                    dev_x, dev_y, activity_names,
                    label="Base HART — float32 TFLite")

    elif args.model == "tflite_int8":
        tflite_path = os.path.join(args.teacher_dir, "HART_int8.tflite")
        eval_tflite(tflite_path, test_x, test_y_oh, test_y_raw,
                    dev_x, dev_y, activity_names,
                    label="Base HART — INT8 quantised TFLite")

    elif args.model == "custom":
        if not args.ckpt:
            raise ValueError("--ckpt is required when using --model custom")
        eval_keras(args.ckpt, input_shape, class_count, meta,
                   dev_x, dev_y, test_x, test_y_oh, test_y_raw,
                   activity_names, label=f"Custom: {args.ckpt}")

    else:
        raise ValueError(f"Unknown --model: {args.model}")


if __name__ == "__main__":
    main()
