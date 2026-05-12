# ENGR6971 — Lightweight HART: Human Activity Recognition Transformer

A lightweight sensor-based Human Activity Recognition (HAR) system built on the HART transformer architecture. The pipeline covers end-to-end training, structured pruning with knowledge distillation, evaluation, and edge deployment preparation for the PAMAP2 dataset.

---

## Results Summary

| Model | Test Accuracy | Weighted F1 | Macro F1 | Size |
|---|---|---|---|---|
| Base HART (best val checkpoint) | 87.28% | 87.00% | 86.43% | 9.14 MB |
| Pruned student (cycle 1, proj=176) | 86.89% | 86.64% | 86.81% | 7.73 MB |
| Compression ratio | — | — | — | **1.18×** |

**Dataset:** PAMAP2 · 9 subjects · 10 activity classes · fold 2 of 5-fold stratified CV (20% test, never seen during training)

---

## Project Structure

```
├── main.py                  # Training entrypoint
├── model.py                 # HART architecture (liteFormer, SensorWiseMHA, DropPath)
├── utils.py                 # Data loading, StratifiedKFold split, dataHolder
├── prune_distill.py         # Iterative pruning + knowledge distillation
├── prune_tracking.py        # Before/after compression metric tracking
├── test_model.py            # Evaluate any saved checkpoint on test fold
├── benchmark_jetson.py      # TFLite latency benchmarking (640 ms budget)
├── intensity_mapper.py      # WHO intensity mapping (MET → light/moderate/vigorous)
├── generate_architecture.py # Pipeline architecture diagram generator
├── datasets/
│   └── DATA_PAMAP2.py       # Gap-aware windowing + LOSO normalisation
├── configs/
│   ├── high_accuracy_90plus.json
│   └── prune_distill_high_accuracy_90plus.json
└── HART_Results/
    └── tracking/            # prune_tracking.csv, comparison plots
```

---

## Pipeline Execution Order

### Step 1 — Preprocess Raw Data
Run once. Reads 9 `.dat` files, applies gap-aware windowing and leave-one-subject-out normalisation, writes `.hkl` files.
```bash
.venv/Scripts/python.exe datasets/DATA_PAMAP2.py
```

### Step 2 — Train Base HART Model
```bash
.venv/Scripts/python.exe main.py \
  --dataset PAMAP2 \
  --config configs/high_accuracy_90plus.json
```

**Outputs:**
```
HART_Results/HART_16frameLength_16TimeStep_192ProjectionSize_0.0005LR_enhancedTok/PAMAP2/
  ├── bestValcheckpoint.weights.h5
  ├── HART.tflite
  ├── HART_int8.tflite
  ├── GlobalACC.csv
  └── run_metadata.json
```

### Step 3 — Record Baseline Metrics
```bash
.venv/Scripts/python.exe prune_tracking.py --stage before \
  --results_dir "HART_Results/HART_16frameLength_16TimeStep_192ProjectionSize_0.0005LR_enhancedTok/PAMAP2"
```

### Step 4 — Prune and Distill
```bash
.venv/Scripts/python.exe prune_distill.py \
  --teacher_ckpt "HART_Results/HART_16frameLength_16TimeStep_192ProjectionSize_0.0005LR_enhancedTok/PAMAP2/bestValcheckpoint.weights.h5" \
  --config configs/prune_distill_high_accuracy_90plus.json \
  --dataset PAMAP2
```

**Cycle progression:**

| Cycle | Projection dim | Notes |
|---|---|---|
| 1 | 176 | **Best** — val acc 88.44% |
| 2 | 160 | |
| 3 | 128 | |
| 4 | 112 | |
| 5 | 96 | |

### Step 5 — Record Post-Compression Metrics
```bash
.venv/Scripts/python.exe prune_tracking.py --stage after \
  --results_dir "HART_Results/pruning"
```

### Step 6 — Evaluate Saved Models
```bash
# Best base model
.venv/Scripts/python.exe test_model.py --model base

# Best pruned student
.venv/Scripts/python.exe test_model.py --model student

# Specific pruning cycle
.venv/Scripts/python.exe test_model.py --model cycle --cycle 1

# Float32 TFLite
.venv/Scripts/python.exe test_model.py --model tflite

# INT8 TFLite
.venv/Scripts/python.exe test_model.py --model tflite_int8

# Any custom checkpoint
.venv/Scripts/python.exe test_model.py --model custom --ckpt path/to/weights.h5
```

### Step 7 — Evaluate Pruned Student Only (no retraining)
```bash
.venv/Scripts/python.exe prune_distill.py \
  --eval_only \
  --teacher_ckpt "HART_Results/.../bestValcheckpoint.weights.h5" \
  --config configs/prune_distill_high_accuracy_90plus.json
```

---

## Architecture

### HART Model
- **6 transformer blocks** — convKernels `[3, 7, 15, 31, 31, 31]`
- **liteFormer** — local attention via depthwise convolutions
- **SensorWiseMHA** — separate attention heads for accelerometer and gyroscope axes
- **DropPath** — stochastic depth with linearly increasing rate per layer
- **Projection dim:** 192 (base) → 176 (pruned best)

### Training Strategy
- **Optimiser:** AdamW with decoupled weight decay
- **Schedule:** Linear warmup + cosine decay (`WarmUpCosineDecay`)
- **EMA:** Exponential Moving Average — decay 0.9999
- **Loss:** CategoricalCrossentropy with `label_smoothing=0.05`
- **Augmentation:** Gaussian jitter N(0, σ × channel\_std) — online, fresh per batch

### Compression Strategy
- **Pruning:** Iterative projection dimension reduction per cycle
- **Distillation:** Hinton-style KL divergence — loss = CE + kd\_weight × T²KL
- **Temperature:** 2.0 · **KD weight:** 0.5
- **Cascading teacher:** each cycle's student becomes the next cycle's teacher

---

## Data Split

```
Per subject — StratifiedKFold(n_splits=5)
├── Folds 0,1,3,4 → Train 80%
│   ├── ~70% Training
│   └── ~10% Validation  (stratified, random_state=1)
└── Fold 2 → Test 20%  ← held out, never used for selection
```

- **Test fold index:** 2 (hardcoded — consistent across all 9 subjects)
- **Test samples:** 4,583
- **Active classes:** 10 (labels 0–8 + label 17 rope\_jumping)

---

## Configuration Files

| Config | Key settings |
|---|---|
| `high_accuracy_90plus.json` | LR 0.0005, proj 192, enhanced tokenizer, EMA 0.9999, 300 epochs, dropout 0.15 |
| `prune_distill_high_accuracy_90plus.json` | T=2.0, KD weight 0.5, LR 1e-4, 5 cycles × 10 epochs |

---

## Tracking Outputs

```
HART_Results/tracking/
  ├── prune_tracking.csv              # Before/after metrics per run
  ├── prune_tracking_graph.png        # Accuracy/F1 over time
  └── prune_before_after_comparison.png
```

---

## Known Issues

| Issue | Detail |
|---|---|
| INT8 TFLite accuracy | 45.7% — calibration failure. Float32 TFLite unaffected (87.3%). |
| `intensity_mapper.py` class 17 | `predict_intensity()` raises `KeyError` on rope\_jumping — not in `ACTIVITY_MET`. Add `"rope_jumping": 8.0` before use. |
| Jetson benchmark | `benchmark_jetson.py` implemented but not yet run on hardware. |
| MixUp | Disabled in `high_accuracy_90plus.json` (`use_mixup: false`). Active only in earlier runs. |

---

## Bug Fixes (21 total across 4 files)

<details>
<summary>Click to expand</summary>

**utils.py** — U-1: mutable class-level lists · U-2: class ref instead of instance · U-3: single-dataset handler

**model.py** — M-1: tensor used as bool · M-2: missing liteFormer attributes · M-3: DropPath never applied · M-4: hardcoded channel slice · M-5: training flag not forwarded · M-6: operator precedence

**main.py** — A-1: duplicate assertion · A-2: silent layer name mismatch · A-3: confusion matrix axes swapped · A-4: IndexError on short version string · A-5: redundant None kwargs

**prune_distill.py** — PD-1: test-set leakage in cycle selection · PD-2: dtype.size on string dtype · PD-3: class count mismatch (10 vs 18) · PD-4: ModelCheckpoint on unbuilt Distiller

**prune_tracking.py** — PT-1: NaN passes `is not None` · PT-2: empty record written · PT-3: no NaN guard on pct_str · PT-4: fold columns excluded from numeric cast · PT-5: missing column crashes groupby

</details>

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `tensorflow==2.20.0` · `scikit-learn` · `hickle` · `pandas` · `numpy` · `matplotlib` · `h5py`

---

## Citation / Course

ENGR6971 — Project and Report I · Winter 2026
