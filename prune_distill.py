import argparse
import csv
import glob
import json
import math
import os
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import model
import utils

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def compute_model_size_mb(m):
    # w.dtype is a plain string in Keras 3 / TF 2.20+ — use np.dtype() to get .itemsize
    total = sum(int(np.prod(w.shape)) * np.dtype(w.dtype).itemsize for w in m.weights)
    return round(total / (1024 * 1024), 4)


def compute_f1_scores(model_obj, x, y_onehot):
    """Return weighted/micro/macro F1 for a compiled or uncompiled model."""
    logits = model_obj(x, training=False)
    y_pred = np.argmax(logits.numpy(), axis=-1)
    y_true = np.argmax(y_onehot, axis=-1)
    weighted = round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 4)
    micro = round(f1_score(y_true, y_pred, average="micro", zero_division=0) * 100, 4)
    macro = round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 4)
    return weighted, micro, macro


def apply_jitter(x, sigma=0.02):
    """Signal-adaptive Gaussian jitter matching main.py augment_sensor_data."""
    signal_std = np.std(x, axis=(0, 1), keepdims=True) + 1e-8
    noise = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return (x + noise * signal_std).astype(np.float32)

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps, min_lr=1e-5):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = int(max(0, warmup_steps))
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        peak = tf.cast(self.peak_lr, tf.float32)
        min_lr = tf.cast(self.min_lr, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        if self.warmup_steps > 0:
            warmup_lr = peak * tf.clip_by_value(step / warmup_steps, 0.0, 1.0)
        else:
            warmup_lr = peak

        cosine_steps = tf.maximum(1.0, total_steps - warmup_steps)
        progress = tf.clip_by_value((step - warmup_steps) / cosine_steps, 0.0, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * progress))
        cosine_lr = min_lr + (peak - min_lr) * cosine_decay
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def build_adamw_optimizer(learning_rate, decay, grad_clipnorm):
    # TF >= 2.11 exposes AdamW at tf.keras.optimizers.AdamW
    adamw_cls = getattr(tf.keras.optimizers, "AdamW", None)
    # TF 2.10 exposes it under the experimental namespace
    if adamw_cls is None:
        adamw_cls = getattr(
            getattr(tf.keras.optimizers, "experimental", None), "AdamW", None
        )
    if adamw_cls is not None:
        return adamw_cls(
            learning_rate=learning_rate,
            weight_decay=decay,
            clipnorm=grad_clipnorm,
        )
    # Fallback: Adam without explicit weight-decay (TF < 2.10)
    print("[Warning] AdamW is not available in this TensorFlow version — "
          "falling back to Adam (no separate weight decay).")
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=grad_clipnorm,
    )



class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=2.0, kd_weight=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.kd_weight = kd_weight
        self.student_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.distill_loss_fn = tf.keras.losses.KLDivergence()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_metric]

    def train_step(self, data):
        x, y = data
        teacher_logits = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            ce_loss = self.student_loss_fn(y, student_logits)
            soft_teacher = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
            soft_student = tf.nn.softmax(student_logits / self.temperature, axis=-1)
            kd_loss = self.distill_loss_fn(soft_teacher, soft_student) * (self.temperature ** 2)
            loss = ce_loss + self.kd_weight * kd_loss

        grads = tape.gradient(loss, self.student.trainable_variables)
        # Filter out None gradients (disconnected nodes / frozen sub-layers)
        grads_and_vars = [
            (g, v) for g, v in zip(grads, self.student.trainable_variables)
            if g is not None
        ]
        self.optimizer.apply_gradients(grads_and_vars)
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y, student_logits)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_metric.result()}

    def test_step(self, data):
        x, y = data
        logits = self.student(x, training=False)
        ce_loss = self.student_loss_fn(y, logits)
        self.loss_tracker.update_state(ce_loss)
        self.acc_metric.update_state(y, logits)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_metric.result()}


class StudentCheckpoint(tf.keras.callbacks.Callback):
    """
    Saves only the inner student model's weights when val_accuracy improves.

    ModelCheckpoint cannot be used on Distiller directly because Keras requires
    the outer wrapper model to be fully built before it can serialise its weights.
    This callback bypasses the wrapper and writes self.model.student weights only,
    which is exactly what the rest of the cycle loop loads back afterwards.
    """
    def __init__(self, filepath, monitor="val_accuracy", mode="max", verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = -np.inf if mode == "max" else np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        improved = (self.mode == "max" and current > self.best) or \
                   (self.mode == "min" and current < self.best)
        if improved:
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved from "
                      f"{self.best:.5f} to {current:.5f} — saving student to {self.filepath}")
            self.best = current
            self.model.student.save_weights(self.filepath)


def read_class_count_from_metadata(teacher_ckpt_path):
    """
    Read activity_count saved by main.py in run_metadata.json.
    The metadata file sits in the same directory as the checkpoint.
    Returns int or None if not found.
    """
    run_dir = os.path.dirname(teacher_ckpt_path)
    meta_path = os.path.join(run_dir, "run_metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        val = meta.get("activity_count")
        return int(val) if val is not None else None
    except Exception as e:
        print(f"[Warning] Could not read activity_count from run_metadata.json: {e}")
        return None


def infer_class_count_from_checkpoint(ckpt_path):
    """
    Fallback: open the h5 checkpoint and return the output-layer class count
    by finding the last Dense kernel's second dimension.
    Returns int or None on failure.
    """
    try:
        import h5py
        kernels = []
        with h5py.File(ckpt_path, "r") as f:
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("kernel"):
                    kernels.append((name, obj.shape))
            f.visititems(_collect)
        if not kernels:
            return None
        kernels.sort(key=lambda x: x[0])
        return int(kernels[-1][1][-1])
    except Exception as e:
        print(f"[Warning] Could not infer class count from checkpoint weights: {e}")
        return None


def load_pamap2(seed, data_dir="./", class_count_override=None):
    data_set_name = "PAMAP2"
    client_count = utils.returnClientByDataset(data_set_name)
    dataset_loader = utils.loadDataset(data_set_name, client_count, "BALANCED", seed, data_dir)
    train_x = dataset_loader.centralTrainData
    train_y = dataset_loader.centralTrainLabel
    test_x = dataset_loader.centralTestData
    test_y = dataset_loader.centralTestLabel

    # FIX W2 – add stratify= to match main.py and preserve class balance in val
    train_x, dev_x, train_y, dev_y = train_test_split(
        train_x, train_y,
        test_size=0.125,
        random_state=seed,
        stratify=train_y,
    )

    detected = len(np.unique(np.hstack((train_y, dev_y, test_y))))
    if class_count_override is not None and class_count_override != detected:
        print(f"[Info] class_count override={class_count_override} "
              f"(detected {detected} unique labels in loaded split — "
              f"some classes absent from fold 2 test set)")
    class_count = class_count_override if class_count_override is not None else detected
    train_y = tf.one_hot(train_y, class_count).numpy().astype(np.float32)
    dev_y = tf.one_hot(dev_y, class_count).numpy().astype(np.float32)
    test_y = tf.one_hot(test_y, class_count).numpy().astype(np.float32)
    return (
        train_x.astype(np.float32), train_y,
        dev_x.astype(np.float32), dev_y,
        test_x.astype(np.float32), test_y,
        class_count,
    )

CYCLE_CSV_HEADER = [
    "cycle", "stage",
    "projection_dim", "filter_heads", "params", "size_mb",
    "val_acc", "test_acc",
    "weighted_f1", "micro_f1", "macro_f1",
    "elapsed_s", "checkpoint",
]


def write_cycle_row(csv_path, row_dict):
    """Append one row to the cycle CSV using DictWriter."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CYCLE_CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def load_config_file(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8-sig") as f:
        if config_path.endswith(".json"):
            return json.load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative structured compression with distillation."
    )
    parser.add_argument("--config", type=str, default="",
                        help="Path to JSON config file (e.g. configs/prune_distill_high_accuracy_90plus.json)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Path to baseline teacher weights (.h5). "
                             "Required unless --eval_only is set.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; evaluate best_pruned_student.weights.h5 "
                             "from --output_dir on the test fold and print full metrics.")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--prune_step", type=float, default=0.10)
    parser.add_argument("--head_prune_step", type=float, default=0.10)
    parser.add_argument("--base_projection_dim", type=int, default=192)
    parser.add_argument("--base_filter_heads", type=int, default=4)
    parser.add_argument("--frame_length", type=int, default=16)
    parser.add_argument("--time_step", type=int, default=16)
    parser.add_argument("--conv_kernels", type=str, default="3,7,15,31,31,31")
    parser.add_argument("--epochs_per_cycle", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd_weight", type=float, default=0.5)
    parser.add_argument("--jitter_sigma", type=float, default=0.02,
                        help="Jitter augmentation sigma (0 = disabled)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--token_dropout", type=float, default=0.1)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--token_based", type=str2bool, default=False)
    parser.add_argument("--enhanced_tokenizer", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str,
                        default="./HART_Results/pruning")
    parser.add_argument("--data_dir", type=str, default="./",
                        help="Root directory containing the datasetStandardized/ folder "
                             "(passed directly to utils.loadDataset)")

    # Load config file first (if given), then let explicit CLI flags override it
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        config_values = load_config_file(pre_args.config)
        parser.set_defaults(**config_values)

    return parser.parse_args()


def build_hart(input_shape, class_count, projection_dim, filter_heads, conv_kernels, args):
    return model.HART(
        input_shape,
        class_count,
        projection_dim=projection_dim,
        patchSize=args.frame_length,
        timeStep=args.time_step,
        filterAttentionHead=filter_heads,
        convKernels=conv_kernels,
        dropout_rate=args.dropout_rate,
        attention_dropout=args.attention_dropout,
        mlp_dropout=args.mlp_dropout,
        token_dropout=args.token_dropout,
        drop_path_rate=args.drop_path_rate,
        useTokens=args.token_based,
        useEnhancedTokenizer=args.enhanced_tokenizer,
    )


def main():
    args = parse_args()

    # ── Upfront validation ────────────────────────────────────────────────────
    if args.eval_only:
        # In eval-only mode we just need the student checkpoint — no teacher required.
        student_ckpt = os.path.join(args.output_dir, "best_pruned_student.weights.h5")
        if not os.path.isfile(student_ckpt):
            cycle_ckpts = sorted(glob.glob(os.path.join(args.output_dir, "student_cycle_*.weights.h5")))
            if cycle_ckpts:
                student_ckpt = cycle_ckpts[-1]
                print(f"[Info] best_pruned_student.weights.h5 not found — "
                      f"using last cycle checkpoint: {student_ckpt}")
            else:
                raise FileNotFoundError(
                    f"\n[Error] No pruned student checkpoint found in {args.output_dir}\n"
                    f"  Run prune_distill.py without --eval_only first to train a student."
                )
        args._eval_student_ckpt = student_ckpt
        # Still need teacher_ckpt to resolve architecture/class-count; check it.
        if args.teacher_ckpt is None:
            found = glob.glob("HART_Results/**/bestValcheckpoint.weights.h5", recursive=True)
            hint = ("\n  Pass --teacher_ckpt so the original architecture can be resolved.\n"
                    + ("  Found:\n" + "".join(f"    {p}\n" for p in found) if found else ""))
            raise ValueError(
                f"\n[Error] --teacher_ckpt is required even with --eval_only "
                f"(needed to resolve model architecture / class count).{hint}"
            )
    else:
        if args.teacher_ckpt is None:
            raise ValueError("--teacher_ckpt is required when not using --eval_only.")

    teacher_ckpt = os.path.abspath(args.teacher_ckpt)
    if not os.path.isfile(teacher_ckpt):
        results_root = os.path.join(os.path.dirname(teacher_ckpt.split("HART_Results")[0]), "HART_Results") \
            if "HART_Results" in teacher_ckpt else "HART_Results"
        found = glob.glob(
            os.path.join(results_root, "**", "bestValcheckpoint.weights.h5"),
            recursive=True,
        )
        hint = ""
        if found:
            hint = "\n  Available checkpoints found:\n" + \
                   "".join(f"    {p}\n" for p in found)
        else:
            hint = (
                "\n  No checkpoints found under HART_Results/.\n"
                "  Run main.py first:\n"
                "    python main.py --config configs/high_accuracy_90plus.json\n"
            )
        raise FileNotFoundError(
            f"\n[Error] Teacher checkpoint not found:\n"
            f"  {teacher_ckpt}\n"
            f"{hint}"
        )
    args.teacher_ckpt = teacher_ckpt
    # ─────────────────────────────────────────────────────────────────────────

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    cycle_csv = os.path.join(args.output_dir, "iterative_prune_distill.csv")
    # Remove stale CSV from a previous run so header is fresh
    if os.path.exists(cycle_csv):
        backup_path = cycle_csv + ".bak"
        os.replace(cycle_csv, backup_path)
        print(f"[Warning] Existing cycle CSV renamed to '{backup_path}' to preserve previous results.")

    # Resolve class count from the teacher run before loading data.
    # Priority: run_metadata.json  →  h5py inspection  →  detected from data split.
    class_count_override = read_class_count_from_metadata(teacher_ckpt)
    if class_count_override is None:
        print("[Info] run_metadata.json missing activity_count — "
              "falling back to h5py checkpoint inspection.")
        class_count_override = infer_class_count_from_checkpoint(teacher_ckpt)
    if class_count_override is not None:
        print(f"[Info] Teacher class count resolved: {class_count_override}")

    train_x, train_y, dev_x, dev_y, test_x, test_y, class_count = load_pamap2(
        args.seed, data_dir=args.data_dir,
        class_count_override=class_count_override,
    )
    input_shape = (train_x.shape[1], train_x.shape[2])
    conv_kernels = [int(k.strip()) for k in args.conv_kernels.split(",") if k.strip()]


    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        # Read the compression summary to find the architecture used for the student
        summary_path = os.path.join(args.output_dir, "compression_summary.json")
        best_proj_dim = args.base_projection_dim
        best_filter_heads = args.base_filter_heads
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            best_proj_dim = summary.get("best_projection_dim", best_proj_dim)
            best_filter_heads = summary.get("best_filter_heads", best_filter_heads)
            print(f"[Eval] Loaded architecture from compression_summary.json: "
                  f"proj={best_proj_dim}  heads={best_filter_heads}")
        else:
            print(f"[Warning] compression_summary.json not found — "
                  f"using base architecture (proj={best_proj_dim}, heads={best_filter_heads}). "
                  f"Results may be wrong if the student used different dimensions.")

        student = build_hart(input_shape, class_count,
                             best_proj_dim, best_filter_heads,
                             conv_kernels, args)
        student.build((None,) + input_shape)
        student.load_weights(args._eval_student_ckpt)
        student.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        _, val_acc  = student.evaluate(dev_x,  dev_y,  verbose=0)
        _, test_acc = student.evaluate(test_x, test_y, verbose=0)
        weighted_f1, micro_f1, macro_f1 = compute_f1_scores(student, test_x, test_y)
        params  = student.count_params()
        size_mb = compute_model_size_mb(student)

        y_pred = np.argmax(student(test_x, training=False).numpy(), axis=-1)
        y_true = np.argmax(test_y, axis=-1)

        print(f"\n{'=' * 60}")
        print(f"  Eval-only results — best pruned student")
        print(f"{'=' * 60}")
        print(f"  Checkpoint  : {args._eval_student_ckpt}")
        print(f"  Architecture: proj={best_proj_dim}  heads={best_filter_heads}")
        print(f"  Parameters  : {params:,}  |  Size: {size_mb:.2f} MB")
        print(f"{'=' * 60}")
        print(f"  Val  accuracy : {val_acc  * 100:.2f}%")
        print(f"  Test accuracy : {test_acc * 100:.2f}%")
        print(f"  Weighted F1   : {weighted_f1:.2f}%")
        print(f"  Micro    F1   : {micro_f1:.2f}%")
        print(f"  Macro    F1   : {macro_f1:.2f}%")
        print(f"  Test samples  : {test_x.shape[0]:,}")
        print(f"{'=' * 60}")

        # Per-class F1
        from sklearn.metrics import classification_report
        print("\n  Per-class F1:")
        report = classification_report(y_true, y_pred, zero_division=0)
        for line in report.splitlines():
            print("  " + line)
        return
    # ─────────────────────────────────────────────────────────────────────────

    if args.jitter_sigma > 0.0:
        aug_x = apply_jitter(train_x, sigma=args.jitter_sigma)
        train_x_aug = np.concatenate([train_x, aug_x], axis=0)
        train_y_aug = np.concatenate([train_y, train_y], axis=0)
        print(f"Jitter augmentation applied (sigma={args.jitter_sigma}). "
              f"Training set: {train_x.shape[0]} -> {train_x_aug.shape[0]} samples.")
    else:
        train_x_aug, train_y_aug = train_x, train_y

    teacher = build_hart(input_shape, class_count,
                         args.base_projection_dim, args.base_filter_heads,
                         conv_kernels, args)
    teacher.build((None,) + input_shape)
    teacher.load_weights(args.teacher_ckpt)
    teacher.trainable = False

    teacher.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    _, baseline_val_acc = teacher.evaluate(dev_x, dev_y, verbose=0)
    _, baseline_test_acc = teacher.evaluate(test_x, test_y, verbose=0)
    baseline_wf1, baseline_mif1, baseline_maf1 = compute_f1_scores(teacher, test_x, test_y)
    baseline_size_mb = compute_model_size_mb(teacher)
    baseline_params = teacher.count_params()

    write_cycle_row(cycle_csv, {
        "cycle": 0,
        "stage": "baseline_teacher",
        "projection_dim": args.base_projection_dim,
        "filter_heads": args.base_filter_heads,
        "params": baseline_params,
        "size_mb": baseline_size_mb,
        "val_acc": round(baseline_val_acc * 100, 4),
        "test_acc": round(baseline_test_acc * 100, 4),
        "weighted_f1": baseline_wf1,
        "micro_f1": baseline_mif1,
        "macro_f1": baseline_maf1,
        "elapsed_s": 0,
        "checkpoint": args.teacher_ckpt,
    })
    print(f"[Baseline] val={baseline_val_acc:.4f}  test={baseline_test_acc:.4f}  "
          f"wF1={baseline_wf1:.2f}%  size={baseline_size_mb:.2f} MB  params={baseline_params:,}")

    # Track best cycle by val_acc (held-out dev set), NOT test_acc.
    # Using test_acc here would be test-set leakage: the test set should
    # never influence which model is selected or saved as "best".
    best_val_acc = -np.inf
    best_checkpoint_path = ""
    best_proj_dim = args.base_projection_dim
    best_filter_heads = args.base_filter_heads

    for cycle in range(1, args.cycles + 1):

        # Compute shrunken dimensions
        projection_factor = max(0.25, 1.0 - args.prune_step * cycle)
        head_factor = max(0.25, 1.0 - args.head_prune_step * cycle)
        projection_dim = max(64, int(round(args.base_projection_dim * projection_factor / 16.0) * 16))
        filter_heads = max(1, int(round(args.base_filter_heads * head_factor)))
        # Enforce (projection_dim // 2) % filter_heads == 0 (liteFormer constraint)
        while (projection_dim // 2) % filter_heads != 0 and filter_heads > 1:
            filter_heads -= 1

        print(f"\n{'=' * 60}")
        print(f"[Cycle {cycle}/{args.cycles}]  proj={projection_dim}  heads={filter_heads}")
        print(f"{'=' * 60}")

        student = build_hart(input_shape, class_count,
                             projection_dim, filter_heads,
                             conv_kernels, args)

        # LR schedule scoped to this cycle's step budget
        steps_per_epoch = max(1, math.ceil(train_x_aug.shape[0] / args.batch_size))
        total_steps = steps_per_epoch * args.epochs_per_cycle
        warmup_steps = int(total_steps * args.warmup_ratio)
        schedule = WarmUpCosineDecay(args.learning_rate, total_steps, warmup_steps,
                                     args.min_learning_rate)
        optimizer = build_adamw_optimizer(schedule, args.weight_decay, args.clipnorm)

        distiller = Distiller(
            student=student, teacher=teacher,
            temperature=args.temperature, kd_weight=args.kd_weight,
        )
        distiller.compile(optimizer=optimizer)

        checkpoint_path = os.path.join(args.output_dir, f"student_cycle_{cycle}.weights.h5")

        checkpoint_cb = StudentCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        )

        patience = max(4, args.epochs_per_cycle // 3)
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            mode="max",
            restore_best_weights=False,  # checkpoint is the single source
            verbose=1,
        )

        start_time = time.time()
        distiller.fit(
            x=train_x_aug, y=train_y_aug,
            validation_data=(dev_x, dev_y),
            batch_size=args.batch_size,
            epochs=args.epochs_per_cycle,
            verbose=1,
            callbacks=[checkpoint_cb, early_stop_cb],
        )
        elapsed = round(time.time() - start_time, 1)

        student.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        if os.path.exists(checkpoint_path):
            student.load_weights(checkpoint_path)
        else:
            # EarlyStopping fired before any epoch improved val_accuracy —
            # evaluate with last-epoch weights and warn.
            print(f"[Warning] Cycle {cycle}: checkpoint was never written "
                  f"(no val_accuracy improvement). Evaluating last-epoch weights.")

        _, val_acc = student.evaluate(dev_x, dev_y, verbose=0)
        _, test_acc = student.evaluate(test_x, test_y, verbose=0)
        weighted_f1, micro_f1, macro_f1 = compute_f1_scores(student, test_x, test_y)
        params = student.count_params()
        size_mb = compute_model_size_mb(student)

        write_cycle_row(cycle_csv, {
            "cycle": cycle,
            "stage": "compressed",
            "projection_dim": projection_dim,
            "filter_heads": filter_heads,
            "params": params,
            "size_mb": size_mb,
            "val_acc": round(float(val_acc) * 100, 4),
            "test_acc": round(float(test_acc) * 100, 4),
            "weighted_f1": weighted_f1,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "elapsed_s": elapsed,
            "checkpoint": checkpoint_path,
        })

        print(f"[Cycle {cycle}] proj={projection_dim}  heads={filter_heads}  "
              f"params={params:,}  size={size_mb:.2f} MB  "
              f"val={val_acc:.4f}  test={test_acc:.4f}  "
              f"wF1={weighted_f1:.2f}%  time={elapsed}s")

        # Track best across all cycles by val_acc (dev set only — no test leakage)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = checkpoint_path
            best_proj_dim = projection_dim
            best_filter_heads = filter_heads

        teacher_ckpt_path = os.path.join(args.output_dir,
                                         f"teacher_cycle_{cycle}.weights.h5")
        student.save_weights(teacher_ckpt_path)

        teacher = build_hart(input_shape, class_count,
                             projection_dim, filter_heads,
                             conv_kernels, args)
        teacher.build((None,) + input_shape)
        teacher.load_weights(teacher_ckpt_path)
        teacher.trainable = False

    if best_checkpoint_path:
        # FIX B2 – rebuild from checkpoint rather than using a stale reference
        best_student = build_hart(input_shape, class_count,
                                  best_proj_dim, best_filter_heads,
                                  conv_kernels, args)
        best_student.build((None,) + input_shape)
        best_student.load_weights(best_checkpoint_path)
        final_path = os.path.join(args.output_dir, "best_pruned_student.weights.h5")
        best_student.save_weights(final_path)

        # Save a JSON summary for easy comparison
        summary = {
            "baseline_test_acc": round(baseline_test_acc * 100, 4),
            "baseline_weighted_f1": baseline_wf1,
            "baseline_params": baseline_params,
            "baseline_size_mb": baseline_size_mb,
            # best cycle was chosen by val_acc (dev set) — no test leakage
            "best_cycle_val_acc": round(best_val_acc * 100, 4),
            "best_projection_dim": best_proj_dim,
            "best_filter_heads": best_filter_heads,
            "best_checkpoint": best_checkpoint_path,
            "final_weights": final_path,
            "compression_ratio": round(baseline_params /
                                       best_student.count_params(), 2),
            "size_reduction_ratio": round(baseline_size_mb /
                                          compute_model_size_mb(best_student), 2),
        }
        summary_path = os.path.join(args.output_dir, "compression_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Compression complete.")
        print(f"  Baseline  test acc : {baseline_test_acc * 100:.2f}%  "
              f"({baseline_params:,} params, {baseline_size_mb:.2f} MB)")
        print(f"  Best student val   : {best_val_acc * 100:.2f}%  "
              f"({best_student.count_params():,} params, "
              f"{compute_model_size_mb(best_student):.2f} MB)")
        print(f"  Compression ratio  : {summary['compression_ratio']}×  "
              f"Size reduction: {summary['size_reduction_ratio']}×")
        print(f"  Saved to           : {final_path}")
        print(f"  Summary JSON       : {summary_path}")
        print(f"  Cycle CSV          : {cycle_csv}")


if __name__ == "__main__":
    main()