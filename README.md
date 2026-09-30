# Lightweight HART — Human Activity Recognition on the Edge

Transformer-based human activity recognition (HAR) from wrist IMU data, trained on **PAMAP2**, compressed with **cascading knowledge distillation**, and prepared for real-time deployment on the **NVIDIA Jetson Orin Nano**. Predictions are mapped to **WHO physical-activity intensity levels** (light / moderate / vigorous) via MET values.

> ENGR 6971 — Project and Report I · Concordia University · Winter 2026
> Author: Ali Awada · Supervisors: Dr. Ahmed Bali, Dr. Paula Lago

---

## Highlights

| | |
|---|---|
| **Best weighted F1** | **88.54 %** (up from 84.62 % early baseline) |
| **Final baseline** | 87.04 % weighted F1 · 86.48 % macro F1 · 9.14 MB |
| **Compressed student** | 86.64 % weighted F1 · 86.81 % macro F1 · 7.73 MB (**1.18× smaller**) |
| **Input** | 128 × 6 window (hand IMU: 3-axis accel + 3-axis gyro, 100 Hz, 1.28 s) |
| **Classes** | 9 PAMAP2 activities → 3 WHO intensity tiers |
| **Base model** | ~2.40 M params · ~36.3 M FLOPs |

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Structure](#repository-structure)
3. [Dataset & Preprocessing](#dataset--preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Compression (Pruning + Distillation)](#compression-pruning--distillation)
7. [Full Pipeline](#full-pipeline)
8. [Results](#results)
9. [Deployment on Jetson](#deployment-on-jetson)
10. [Known Issues & Limitations](#known-issues--limitations)
11. [Roadmap](#roadmap)
12. [References](#references)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Alii-Awada/Lightweight-HART-Transformer.git
cd Lightweight-HART-Transformer

# 2. Environment (Python 3.9–3.10)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # TensorFlow 2.10.1

# 3. Train with the best-performing config
python main.py --dataset PAMAP2 --config configs/high_accuracy_90plus.json

# 4. Evaluate the saved model on the held-out test fold
python test_model.py --model base
```

Preprocessed `.hkl` files are already included in `datasetStandardized/PAMAP2/`, so you can train without downloading raw data. To regenerate them, see [Dataset & Preprocessing](#dataset--preprocessing).

> **Note:** all scripts use paths relative to the repository root — run them from there.

---

## Repository Structure

```
Lightweight-HART-Transformer/
├── main.py                 # Training entrypoint (AdamW, cosine LR, EMA, jitter/MixUp, TFLite export)
├── model.py                # HART architecture: SensorWiseMHA, liteFormer, DropPath, multi-scale tokenizer
├── utils.py                # Data loading, per-subject StratifiedKFold split
├── prune_distill.py        # Iterative dimension reduction + cascading knowledge distillation
├── prune_tracking.py       # Before/after compression metric logging and plots
├── test_model.py           # Evaluate any checkpoint / TFLite model on the test fold
├── benchmark_jetson.py     # TFLite latency & throughput benchmark (640 ms real-time budget)
├── intensity_mapper.py     # Activity → MET → WHO intensity + weekly moderate-equivalent minutes
├── requirements.txt
├── configs/
│   ├── baseline.json                           # Original HART-style recipe (MixUp on)
│   ├── high_accuracy_90plus.json               # Best training recipe
│   └── prune_distill_high_accuracy_90plus.json # Compression recipe
├── datasets/
│   └── DATA_PAMAP2.py      # Raw .dat → gap-aware windows → .hkl
├── datasetStandardized/
│   └── PAMAP2/             # UserData{0..8}.hkl, UserLabel{0..8}.hkl, activity_labels.txt
└── HART_Results/
    ├── HART_<config>/PAMAP2/   # Per-run checkpoints, metrics, plots (git-ignored)
    ├── pruning/                # Student checkpoints + compression_summary.json (git-ignored)
    └── tracking/               # prune_tracking.csv + comparison plots
```

---

## Dataset & Preprocessing

**PAMAP2** — 9 subjects, multiple body-worn IMUs at 100 Hz ([UCI repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)). Only the **hand IMU** (accelerometer + gyroscope, 6 channels) is used.

| Idx | Activity | MET | WHO tier |
|---|---|---|---|
| 0 | Lying | 1.0 | Light |
| 1 | Sitting | 1.3 | Light |
| 2 | Standing | 1.5 | Light |
| 3 | Walking | 3.5 | Moderate |
| 4 | Running | 8.0 | Vigorous |
| 5 | Cycling | 6.0 | Vigorous |
| 6 | Nordic walking | 4.8 | Moderate |
| 7 | Ascending stairs | 4.0 | Moderate |
| 8 | Descending stairs | 3.5 | Moderate |

WHO thresholds: light < 3.0 MET · moderate 3.0–5.99 MET · vigorous ≥ 6.0 MET.

### Regenerating the `.hkl` files

Place the raw dataset so the `Protocol/` folder sits **next to** the repository folder:

```
parent/
├── PAMAP2_Dataset/Protocol/subject101.dat ... subject109.dat
└── Lightweight-HART-Transformer/
```

Then, from the repo root:

```bash
python datasets/DATA_PAMAP2.py
```

### Pipeline steps (`DATA_PAMAP2.py`)

1. **Column selection** — timestamp, activity label, 6 IMU channels.
2. **NaN removal** — `dropna()` before windowing.
3. **Activity filtering** — keep the 9 core activities; drop transient/optional ones.
4. **Gap-aware windowing** — any inter-sample gap > 25 ms splits the recording into contiguous chunks; each chunk is windowed independently (128 samples, step 64, 50 % overlap). Windows are kept only if all 128 samples share one label.
5. **Leave-one-subject-out normalisation** — each subject is z-scored with mean/std from the *other 8* subjects (no cross-subject leakage).
6. **Serialisation** — `.hkl` files + `activity_labels.txt`.

### Train / validation / test split (`utils.py`)

```
Per subject — StratifiedKFold(n_splits=5, shuffle=False)
├── Folds 0,1,3,4 → 80 %
│   ├── ~70 % Train
│   └── ~10 % Validation   (stratified 12.5 % of the train pool)
└── Fold 2       → 20 % Test  (never used for model selection)
```

Per-subject partitions are concatenated into `centralTrainData` / `centralTestData`.

---

## Model Architecture

HART (Ek et al., 2023) processes accelerometer and gyroscope features in separate attention subspaces.

```
Input (128, 6)
  → Multi-scale Conv1D tokenizer   8 tokens × 192 dim  (kernels 8 / 16 / 31)
  → Learnable positional embedding
  → 6 × HART block                 conv kernels [3, 7, 15, 31, 31, 31]
        ├── AccMHA      (accelerometer attention)
        ├── GyroMHA     (gyroscope attention)
        ├── LiteFormer  (depthwise temporal conv)
        ├── GlobalMHA   (cross-modal attention)
        └── LayerNorm · MLP · residuals · DropPath (stochastic depth)
  → Global average pooling
  → MLP head (Dense 1024 → Swish → Dropout) → softmax (9 classes)
  → MET lookup → Light / Moderate / Vigorous
```

---

## Training

```bash
python main.py --dataset PAMAP2 --config configs/high_accuracy_90plus.json
```

Any config key can be overridden on the command line, e.g. `--learningRate 0.0003 --localEpoch 200`.

| Technique | Setting (high-accuracy config) |
|---|---|
| Optimiser | AdamW, weight decay 0.01, clipnorm 1.0 |
| LR schedule | Linear warmup (10 %) → cosine decay, peak 5e-4 → 5e-6 |
| Epochs / batch | 300 / 128, early stopping (patience 25) |
| EMA | decay 0.9999, best-val EMA weights restored after training |
| Loss | Categorical cross-entropy, label smoothing 0.05 |
| Regularisation | DropPath 0.12, dropout 0.15, attention dropout 0.08 |
| Augmentation | Online `tf.data` Gaussian jitter (σ = 0.015 × channel std), fresh per batch |
| MixUp | Disabled (enabled at α = 0.2 in `baseline.json`) |
| Class weights | On when MixUp is off |

### Config comparison

| Parameter | `baseline.json` | `high_accuracy_90plus.json` | Effect |
|---|---|---|---|
| Peak LR | 1e-3 | **5e-4** | More stable convergence |
| Warmup ratio | 0.06 | **0.10** | Smoother entry into cosine decay |
| EMA decay | 0.999 | **0.9999** | Smoother, better final weights |
| DropPath | 0.10 | **0.12** | Slightly stronger regularisation |
| MixUp | α = 0.2 | **off** | Sharper class boundaries |
| Batch size | 256 | **128** | More gradient noise, better generalisation |
| Epochs | 200 | **300** | — |

### Outputs

```
HART_Results/HART_16frameLength_16TimeStep_192ProjectionSize_0.0005LR_enhancedTok/PAMAP2/
├── bestValcheckpoint.weights.h5   # model selection by val accuracy
├── HART.tflite                    # float32 TFLite
├── HART_int8.tflite               # INT8 post-training quantised
├── GlobalACC.csv                  # train / val / test F1 summary
├── history.hkl                    # training curves
└── run_metadata.json              # hyperparameters, seed, versions, dataset checksums
```

---

## Compression (Pruning + Distillation)

Instead of unstructured weight pruning (irregular sparsity, poor edge speed-ups), each cycle builds a **smaller dense HART** and trains it by distilling from the previous cycle's model.

```bash
python prune_distill.py \
  --teacher_ckpt "HART_Results/HART_16frameLength_16TimeStep_192ProjectionSize_0.0005LR_enhancedTok/PAMAP2/bestValcheckpoint.weights.h5" \
  --config configs/prune_distill_high_accuracy_90plus.json
```

- **Projection dim per cycle:** `round(192 × max(0.25, 1 − 0.10·c) / 16) × 16` → 176 · 160 · 128 · 112 · 96 (attention heads are reduced alongside)
- **Loss:** `CE(student, y) + 0.5 · T² · KL(teacher_T ‖ student_T)`, T = 2.0
- **Cascading:** each cycle's student becomes the next cycle's teacher
- **Training:** 5 cycles × 10 epochs, LR 1e-4, jitter only (MixUp off — KD soft targets already regularise)
- **Selection:** best cycle chosen by **validation accuracy only** (never test accuracy)

| Cycle | Proj dim | Size | Val acc |
|---|---|---|---|
| Teacher | 192 | 9.14 MB | 87.30 % |
| **1 ★** | **176** | **7.73 MB** | **88.44 %** |
| 2–5 | 160 → 96 | smaller | decreasing (~86 % → ~80 %) |

Evaluate existing students without retraining:

```bash
python prune_distill.py --eval_only --teacher_ckpt <path> --config configs/prune_distill_high_accuracy_90plus.json
```

Outputs go to `HART_Results/pruning/` (`student_cycle_*.weights.h5`, `teacher_cycle_*.weights.h5`, `best_pruned_student.weights.h5`, `compression_summary.json`).

---

## Full Pipeline

Run in this order — each step depends on the previous one.

```bash
# 1. Preprocess (once)
python datasets/DATA_PAMAP2.py

# 2. Train baseline
python main.py --dataset PAMAP2 --config configs/high_accuracy_90plus.json

# 3. Record baseline metrics  ← must run BEFORE step 4
python prune_tracking.py --stage before --run-dir "HART_Results/<run_dir>/PAMAP2"

# 4. Compress
python prune_distill.py --teacher_ckpt "HART_Results/<run_dir>/PAMAP2/bestValcheckpoint.weights.h5" \
                        --config configs/prune_distill_high_accuracy_90plus.json

# 5. Record compressed metrics (appends row, regenerates plots)
python prune_tracking.py --stage after --run-dir "HART_Results/pruning"

# 6. Evaluate
python test_model.py --model base          # best-val base model
python test_model.py --model student       # best pruned student
python test_model.py --model cycle --cycle 3
python test_model.py --model tflite        # float32 TFLite
python test_model.py --model tflite_int8   # INT8 TFLite
python test_model.py --model custom --ckpt path/to/weights.h5

# 7. Benchmark on the Jetson (see below)
```

To iterate on a new config, repeat steps 2 → 5. `prune_tracking.csv` is append-only, so history accumulates across experiments. Regenerate plots only with `python prune_tracking.py --plot-only`.

---

## Results

### Training progression (test fold, weighted F1)

| Date | Config | Train acc | Weighted F1 | Micro F1 | Macro F1 | Size |
|---|---|---|---|---|---|---|
| 2026-02-11 | LR 5e-3 (early baseline) | 85.34 | 84.62 | 84.62 | 82.23 | — |
| 2026-03-03 | LR 1e-3 | 99.56 | 84.96 | 85.22 | 83.36 | — |
| 2026-03-10 | LR 7e-4, enhanced tokenizer | 97.63 | 85.41 | 85.84 | 83.75 | 9.10 MB |
| 2026-03-10 | LR 5e-4, enhanced tokenizer | 100.0 | 87.51 | 87.75 | 85.97 | 9.14 MB |
| 2026-03-16 | LR 5e-4, enhanced tokenizer | 99.70 | **88.54** | **88.82** | **88.23** | 9.14 MB |
| 2026-03-26 | LR 5e-4, enhanced tokenizer (final) | 99.08 | 87.04 | 87.19 | 86.48 | 9.14 MB |
| 2026-04-01 | Pruned student (KD, proj 176) | — | 86.64 | 86.89 | 86.81 | 7.73 MB |

Largest gains came from the **multi-scale tokenizer**, a **lower peak LR**, and a **higher EMA decay**. Near-100 % training accuracy shows the model can overfit, which is why EMA, stochastic depth and jitter matter. Run-to-run variance between identical configs is ~1.5 F1 points — use multiple seeds when comparing changes.

Plots: `HART_Results/tracking/prune_tracking_graph.png`, `HART_Results/tracking/prune_before_after_comparison.png`.

---

## Deployment on Jetson

**Real-time budget:** a new window arrives every 64 samples at 100 Hz → **640 ms per inference**. A model is real-time capable if its **P95 latency** stays below that.

On the Jetson Orin Nano:

```python
from benchmark_jetson import benchmark_tflite
benchmark_tflite("HART_int8.tflite", input_shape=(128, 6))   # 50 warm-up + 1000 timed runs
```

Measure power in parallel:

```bash
sudo tegrastats --interval 100 --logfile power.log
```

Reports mean latency, P95 latency, throughput (inferences/s) and a real-time pass/fail.

### Intensity mapping

```python
from intensity_mapper import predict_intensity, who_weekly_minutes

intensities, mets = predict_intensity(activity_probs, activity_labels)
summary = who_weekly_minutes(intensities, window_step_sec=0.64)
```

Maps predicted classes → MET → Light / Moderate / Vigorous, aggregates minutes over time, and computes **moderate-equivalent minutes** (vigorous counts double) for comparison against WHO's 150 min/week target.

---

## Known Issues & Limitations

| Issue | Detail |
|---|---|
| INT8 TFLite accuracy | ~45.7 % — post-training quantisation calibration is failing. Float32 TFLite matches Keras (~87.3 %). Use float32 until fixed. |
| Jetson benchmark | `benchmark_jetson.py` is implemented but has not yet been run on hardware. |
| Evaluation protocol | Per-subject 5-fold split (not LOSO), `shuffle=False` → contiguous temporal folds; 50 % window overlap can leak near-duplicates across the train/test boundary. Scores may be slightly optimistic for unseen users. |
| Compression ratio | Only cycle 1 (1.18×) is retained at near-baseline accuracy; deeper cycles lose 2–8 val-accuracy points. |
| Tracking plot | Some early before/after rows point to the same run directory, so their bars are identical. |

---

## Roadmap

- [ ] Fix INT8 calibration (larger / class-balanced representative set, or quantisation-aware training)
- [ ] Run `benchmark_jetson.py` on the Orin Nano; add TensorRT (FP16/INT8) export and results
- [ ] Leave-one-subject-out evaluation for true cross-user generalisation
- [ ] Multi-seed runs with mean ± std reporting
- [ ] Longer KD schedules per cycle to recover accuracy at 128/112 dims
- [ ] Additional IMU placements (chest, ankle) and heart-rate fusion

---

## References

- Ek, S. et al. (2023). *Transformer-based models to deal with heterogeneous environments in Human Activity Recognition.* Personal and Ubiquitous Computing, 27, 2267–2280. [doi:10.1007/s00779-023-01776-3](https://doi.org/10.1007/s00779-023-01776-3)
- Reiss, A. (2012). *PAMAP2 Physical Activity Monitoring.* UCI Machine Learning Repository.
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network.* NIPS Deep Learning Workshop.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR.
- Huang, G. et al. (2016). *Deep networks with stochastic depth.* ECCV.
- Zhang, H. et al. (2018). *MixUp: Beyond empirical risk minimization.* ICLR.
- Touvron, H. et al. (2021). *Training data-efficient image transformers & distillation through attention.* ICML.
- Bulling, A. et al. (2014). *A tutorial on human activity recognition using body-worn inertial sensors.* ACM Computing Surveys.
- World Health Organization (2020). *WHO guidelines on physical activity and sedentary behaviour.*
- NVIDIA (2023). *Jetson Orin Nano product brief.*
