#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
randomSeed = 1
os.environ['PYTHONHASHSEED']=str(randomSeed)


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


# In[ ]:


import csv
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import time
import hickle as hkl 
import random
import math
import logging
import shutil
import gc
import sys
import sklearn.manifold
import seaborn as sns
import argparse
import matplotlib.gridspec as gridspec
import __main__ as main
import json
import hashlib
import platform


# In[ ]:


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


def load_config_file(config_path):
    if not config_path:
        return {}
    # Use utf-8-sig to transparently handle files saved with UTF-8 BOM.
    with open(config_path, "r", encoding="utf-8-sig") as f:
        if config_path.endswith(".json"):
            return json.load(f)
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to load YAML configs.") from exc
            return yaml.safe_load(f) or {}
    raise ValueError("Unsupported config format. Use .json, .yaml, or .yml")


# In[ ]:


def configure_devices():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            tf.config.set_visible_devices([], "GPU")
            return "cpu"
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return "gpu"
    except Exception:
        # Fall back to CPU if GPU setup fails.
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "cpu"


device_mode = configure_devices()


def compute_model_flops(model, input_shape_no_batch):
    try:
        input_dtype = model.inputs[0].dtype if model.inputs else tf.float32
        concrete_func = tf.function(model).get_concrete_function(
            tf.TensorSpec([1] + list(input_shape_no_batch), dtype=input_dtype)
        )
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            flops = tf.compat.v1.profiler.profile(
                graph=graph,
                options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(),
            )
        return int(flops.total_float_ops) if flops is not None else None
    except Exception as exc:
        print(f"Could not compute FLOPs: {exc}")
        return None


def compute_model_size_mb(model):
    def _dtype_nbytes(dtype_obj):
        try:
            return int(tf.as_dtype(dtype_obj).size)
        except Exception:
            return int(np.dtype(dtype_obj).itemsize)

    total_bytes = int(
        sum(int(np.prod(v.shape)) * _dtype_nbytes(v.dtype) for v in model.weights)
    )
    return total_bytes / (1024.0 * 1024.0)


# In[ ]:


# which CPU/GPU to use
# "-1,0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set The Default Hyperparameters Here

architecture = "HART"
# MobileHART, HART

# PAMAP2
dataSetName = 'PAMAP2'

#BALANCED, UNBALANCED
dataConfig = "BALANCED"

# Show training verbose: 0,1
showTrainVerbose = 1

# input window size 
segment_size = 128

# input channel count (auto-detected after dataset load)
num_input_channels = None

learningRate = 1e-3

# model drop out rate
dropout_rate = 0.2

# local epoch
localEpoch = 200
# or 4 
frameLength = 16

timeStep = 16

positionDevice = ''
tokenBased = False
hartEnhancedTokenizer = False
config_path = ''

measureEnergy = False

# Optimizer and schedule defaults (Transformer-friendly).
weight_decay = 0.01
warmup_ratio = 0.06
min_learning_rate = 1e-5
clipnorm = 1.0

# Regularization defaults.
attention_dropout = 0.1
mlp_dropout = 0.1
token_dropout = 0.1
drop_path_rate = 0.1

# EMA + augmentation defaults.
use_ema = True
ema_decay = 0.999
use_jitter = True
jitter_sigma = 0.02
use_mixup = True
mixup_alpha = 0.2


# In[ ]:


# hyperparameter for the model

batch_size = 256
projection_dim = 192
filterAttentionHead = 4

# To adjust the number of blocks in for HART, Add or remove the conv kernel size here
# each conv kernel length is for one HART block
convKernels = [3, 7, 15, 31, 31, 31]


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
            warmup_progress = tf.clip_by_value(step / warmup_steps, 0.0, 1.0)
            warmup_lr = peak * warmup_progress
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


class EMACallback(tf.keras.callbacks.Callback):
    def __init__(self, decay=0.999, monitor="val_accuracy"):
        super().__init__()
        self.decay = float(decay)
        self.monitor = monitor
        self.shadow_weights = None
        self.best_shadow_weights = None
        self.best_metric = float("-inf")

    def on_train_begin(self, logs=None):
        self.shadow_weights = [w.numpy().copy() for w in self.model.trainable_weights]
        self.best_shadow_weights = [w.copy() for w in self.shadow_weights]

    def on_train_batch_end(self, batch, logs=None):
        if self.shadow_weights is None:
            return
        current_weights = self.model.trainable_weights
        for i, weight in enumerate(current_weights):
            self.shadow_weights[i] = self.decay * self.shadow_weights[i] + (1.0 - self.decay) * weight.numpy()

    def on_epoch_end(self, epoch, logs=None):
        if self.shadow_weights is None:
            return
        metric_val = float(logs.get(self.monitor, -np.inf)) if logs else -np.inf
        if metric_val > self.best_metric:
            self.best_metric = metric_val
            self.best_shadow_weights = [w.copy() for w in self.shadow_weights]

    def apply_best_ema(self):
        if self.best_shadow_weights is None:
            return False
        self.model.set_weights(self.best_shadow_weights)
        return True


def compute_dataset_checksum(dataset_dir):
    digest = hashlib.sha256()
    for root, _, files in os.walk(dataset_dir):
        for file_name in sorted(files):
            if not file_name.endswith(".hkl"):
                continue
            file_path = os.path.join(root, file_name)
            digest.update(file_name.encode("utf-8"))
            digest.update(str(os.path.getsize(file_path)).encode("utf-8"))
            digest.update(str(int(os.path.getmtime(file_path))).encode("utf-8"))
    return digest.hexdigest()


def save_run_metadata(filepath, metadata_dict):
    with open(os.path.join(filepath, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, sort_keys=True)


def build_adamw_optimizer(learning_rate, decay, grad_clipnorm):
    adamw_cls = getattr(tf.keras.optimizers, "AdamW", None)
    if adamw_cls is None:
        adamw_cls = tf.keras.optimizers.experimental.AdamW
    return adamw_cls(
        learning_rate=learning_rate,
        weight_decay=decay,
        clipnorm=grad_clipnorm,
    )


def make_dataset(
    x,
    y,
    batch_size,
    augment=False,
    use_jitter_aug=False,
    jitter_std=0.02,
    use_mixup_aug=False,
    mixup_alpha_value=0.2,
):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if augment:
        signal_std = tf.constant(
            (np.std(x, axis=(0, 1), keepdims=True).astype(np.float32) + 1e-8)
        )

        def apply_jitter(xb, yb):
            if not use_jitter_aug:
                return xb, yb
            noise = tf.random.normal(tf.shape(xb), dtype=xb.dtype) * tf.cast(jitter_std, xb.dtype)
            return xb + noise * signal_std, yb

        def apply_mixup(xb, yb):
            if not use_mixup_aug or mixup_alpha_value <= 0.0:
                return xb, yb
            alpha = tf.cast(mixup_alpha_value, xb.dtype)
            gamma_a = tf.random.gamma([tf.shape(xb)[0], 1, 1], alpha=alpha, dtype=xb.dtype)
            gamma_b = tf.random.gamma([tf.shape(xb)[0], 1, 1], alpha=alpha, dtype=xb.dtype)
            lam = gamma_a / (gamma_a + gamma_b + tf.cast(1e-8, xb.dtype))
            idx = tf.random.shuffle(tf.range(tf.shape(xb)[0]))
            mixed_x = lam * xb + (1.0 - lam) * tf.gather(xb, idx)
            lam_y = lam[:, :, 0]
            mixed_y = lam_y * yb + (1.0 - lam_y) * tf.gather(yb, idx)
            return mixed_x, mixed_y

        ds = (
            ds.shuffle(min(10000, int(x.shape[0])), reshuffle_each_iteration=True)
            .batch(batch_size, drop_remainder=True)
            .map(apply_jitter, num_parallel_calls=tf.data.AUTOTUNE)
            .map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)
        )
    else:
        ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def export_tflite_int8(model, representative_data, filepath):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(0, min(500, len(representative_data)), 1):
            sample = representative_data[i:i + 1].astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"INT8 TFLite saved: {filepath}  ({size_mb:.3f} MB)")
    return size_mb


def print_final_test_summary(
    test_accuracy,
    weighted_f1,
    micro_f1,
    macro_f1,
    confusion_matrix_df,
    global_metrics_path,
    confusion_matrix_path,
    complexity_matrix_path,
    tflite_path,
    test_fold_index=None,
    n_folds=None,
    test_split_pct=None,
    y_true=None,
    y_pred=None,
    activity_labels=None,
    n_test_samples=None,
):
    W = 62
    print("\n" + "=" * W)

    # ── Fold / split headline ─────────────────────────────────────────────
    if test_fold_index is not None and n_folds is not None:
        pct_str = f"  ({test_split_pct}% of data)" if test_split_pct is not None else ""
        samples_str = f"  |  {n_test_samples:,} samples" if n_test_samples is not None else ""
        print(f"  Test Fold Results  —  fold {test_fold_index} of {n_folds}{pct_str}{samples_str}")
    else:
        print("  Final Test Results")
    print("=" * W)

    # ── Overall metrics ───────────────────────────────────────────────────
    print(f"  {'Metric':<20} {'Value':>10}")
    print("-" * W)
    print(f"  {'Accuracy':<20} {round(test_accuracy * 100, 2):>9.2f}%")
    print(f"  {'Weighted F1':<20} {round(weighted_f1 * 100, 2):>9.2f}%")
    print(f"  {'Micro F1':<20} {round(micro_f1 * 100, 2):>9.2f}%")
    print(f"  {'Macro F1':<20} {round(macro_f1 * 100, 2):>9.2f}%")

    # ── Per-class F1 for this test fold ───────────────────────────────────
    if y_true is not None and y_pred is not None and activity_labels is not None:
        from sklearn.metrics import f1_score as _f1
        per_class_f1 = _f1(y_true, y_pred, average=None,
                           labels=list(range(len(activity_labels))),
                           zero_division=0)
        print("\n" + "-" * W)
        print(f"  Per-class F1  (fold {test_fold_index} test set)")
        print("-" * W)
        print(f"  {'Class':<24} {'F1 (%)':>8}  {'Bar'}")
        print("-" * W)
        for cls_idx, (label, f1_val) in enumerate(
            zip(activity_labels, per_class_f1)
        ):
            bar_len = int(round(f1_val * 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {label:<24} {f1_val * 100:>7.2f}%  {bar}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  Confusion matrix (rows = Ground Truth, cols = Prediction)")
    print("-" * W)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
    ):
        print(confusion_matrix_df)

    # ── Saved artifacts ───────────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  Saved artifacts:")
    print(f"    Metrics CSV       : {global_metrics_path}")
    print(f"    Confusion matrix  : {confusion_matrix_path}")
    print(f"    Complexity matrix : {complexity_matrix_path}")
    print(f"    INT8 TFLite       : {tflite_path}")
    print("=" * W + "\n")


# In[ ]:


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--config', type=str, default='',
                        help='Path to JSON/YAML config file')
    parser.add_argument('--batch_size', type=int, default=batch_size, 
                        help='Batch size of the training')  
    parser.add_argument('--localEpoch', type=int, default=localEpoch, 
                        help='Number of epochs for training')  
    parser.add_argument('--architecture', type=str, default=architecture, 
                        help='Choose between HART or MobileHART')  
    parser.add_argument('--projection_dim', type=int, default=projection_dim, 
                        help='Size of the projection dimensions')  
    parser.add_argument('--frame_length', type=int, default=frameLength, 
                help='Patch Size')  
    parser.add_argument('--time_step', type=int, default=timeStep, 
            help='Stride Size')  
    parser.add_argument('--dataset', type=str, default=dataSetName, 
        help='Dataset')  
    parser.add_argument('--learningRate', type=float, default=learningRate,
        help='Peak learning rate for optimizer')
    parser.add_argument('--dropout_rate', type=float, default=dropout_rate,
        help='Dropout rate for model blocks')
    parser.add_argument('--weight_decay', type=float, default=weight_decay,
        help='Decoupled weight decay for AdamW')
    parser.add_argument('--warmup_ratio', type=float, default=warmup_ratio,
        help='Warmup ratio for warmup + cosine schedule')
    parser.add_argument('--min_learning_rate', type=float, default=min_learning_rate,
        help='Minimum learning rate floor for cosine decay')
    parser.add_argument('--clipnorm', type=float, default=clipnorm,
        help='Gradient clipping norm')
    parser.add_argument('--attention_dropout', type=float, default=attention_dropout,
        help='Attention dropout rate')
    parser.add_argument('--mlp_dropout', type=float, default=mlp_dropout,
        help='MLP dropout rate')
    parser.add_argument('--token_dropout', type=float, default=token_dropout,
        help='Token dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=drop_path_rate,
        help='Maximum stochastic depth rate')
    parser.add_argument('--use_ema', type=str2bool, default=use_ema,
        help='Enable EMA of model weights')
    parser.add_argument('--ema_decay', type=float, default=ema_decay,
        help='EMA decay factor')
    parser.add_argument('--use_jitter', type=str2bool, default=use_jitter,
        help='Enable Gaussian jitter augmentation')
    parser.add_argument('--jitter_sigma', type=float, default=jitter_sigma,
        help='Jitter sigma as fraction of signal std')
    parser.add_argument('--use_mixup', type=str2bool, default=use_mixup,
        help='Enable MixUp augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=mixup_alpha,
        help='MixUp alpha parameter')
    parser.add_argument('--tokenBased', type=str2bool, default=tokenBased, 
        help='Use Token or Global Average Pooling')  
    parser.add_argument('--hartEnhancedTokenizer', type=str2bool, default=hartEnhancedTokenizer,
        help='Use enhanced tokenizer + global attention fusion in HART')
    parser.add_argument('--positionDevice', type=str, default=positionDevice, 
        help='Test is done other position not in training, if empty, uses a 70/10/20 train/dev/test ratio ')  

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        config_values = load_config_file(pre_args.config)
        parser.set_defaults(**config_values)

    args = parser.parse_args()
    return args


# In[ ]:


def is_interactive():
    return not hasattr(main, '__file__')


# In[ ]:


if not is_interactive():
    args = add_fit_args(argparse.ArgumentParser(description='Human Activity Recognition Transformer'))
    localEpoch = args.localEpoch
    batch_size = args.batch_size
    architecture = args.architecture
    projection_dim = args.projection_dim
    frameLength = args.frame_length
    timeStep = args.time_step
    dataSetName = args.dataset
    learningRate = args.learningRate
    dropout_rate = args.dropout_rate
    weight_decay = args.weight_decay
    warmup_ratio = args.warmup_ratio
    min_learning_rate = args.min_learning_rate
    clipnorm = args.clipnorm
    attention_dropout = args.attention_dropout
    mlp_dropout = args.mlp_dropout
    token_dropout = args.token_dropout
    drop_path_rate = args.drop_path_rate
    use_ema = args.use_ema
    ema_decay = args.ema_decay
    use_jitter = args.use_jitter
    jitter_sigma = args.jitter_sigma
    use_mixup = args.use_mixup
    mixup_alpha = args.mixup_alpha
    tokenBased = args.tokenBased
    hartEnhancedTokenizer = args.hartEnhancedTokenizer
    positionDevice = args.positionDevice
    config_path = args.config
    
input_shape = None
projectionHalf = projection_dim//2
projectionQuarter = projection_dim//4

transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers

R = projectionHalf // filterAttentionHead
assert R * filterAttentionHead == projectionHalf


segmentTime = [x for x in range(0,segment_size - frameLength + timeStep,timeStep)]
if(positionDevice != ''):
    assert dataSetName == "RealWorld" or dataSetName == "HHAR"


# In[ ]:


# specifying activities and where the results will be stored 
ACTIVITY_LABEL = [
    'Lying',
    'Sitting',
    'Standing',
    'Walking',
    'Running',
    'Cycling',
    'Nordic_walking',
    'Watching_tv',
    'Computer_work',
    'Car_driving',
    'Ascending_stairs',
    'Descending_stairs',
    'Vacuum_cleaning',
    'Ironing',
    'Folding_laundry',
    'House_cleaning',
    'Playing_soccer',
    'Rope_jumping',
]

activityCount = len(ACTIVITY_LABEL)

architectureType = str(architecture)+'_'+str(int(frameLength))+'frameLength_'+str(timeStep)+'TimeStep_'+str(projection_dim)+"ProjectionSize_"+str(learningRate)+'LR'
if(tokenBased):
    architectureType = architectureType + "_tokenBased"
if(architecture == "HART" and hartEnhancedTokenizer):
    architectureType = architectureType + "_enhancedTok"
    
if(positionDevice != ''):
    architectureType = architectureType + "_PositionWise_" + str(positionDevice)
mainDir = './'

if(localEpoch < 20):
    architectureType =  "Tests/"+str(architectureType)
filepath = mainDir +'HART_Results/'+architectureType+'/'+dataSetName+'/'
os.makedirs(filepath, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

attentionPath = filepath+"attentionImages/"
os.makedirs(attentionPath, exist_ok=True)

bestModelPath = filepath + 'bestModels/'
os.makedirs(bestModelPath, exist_ok=True)

currentModelPath = filepath + 'currentModels/'
os.makedirs(currentModelPath, exist_ok=True)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
np.random.seed(randomSeed)
tf.keras.utils.set_random_seed(randomSeed)
tf.random.set_seed(randomSeed)
random.seed(randomSeed)


# In[ ]:

if(dataSetName == "COMBINED"):
    datasetList = ["UCI","RealWorld","HHAR", "MotionSense","SHL_128"]
    ACTIVITY_LABEL = ['Walk', 'Upstair', 'Downstair', 'Sit', 'Stand', 'Lay', 'Jump','Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
    activityCount = len(ACTIVITY_LABEL)
    UCI = [0,1,2,3,4,5]
    REALWORLD_CLIENT = [2,1,6,5,7,3,4,0]
    HHAR = [3,4,0,1,2,8]
    MotionSense = [2,1,3,4,0,7]
    SHL = [4,0,7,8,9,10,11,12]

    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    testFoldInfoList = []
    for dataSetName in datasetList:
        clientCount = utils.returnClientByDataset(dataSetName)
        loadedDataset = utils.loadDataset(dataSetName,clientCount,dataConfig,randomSeed,mainDir)
        centralTrainData.append(loadedDataset.centralTrainData)
        centralTrainLabel.append(loadedDataset.centralTrainLabel)
        centralTestData.append(loadedDataset.centralTestData)
        centralTestLabel.append(loadedDataset.centralTestLabel)
        testFoldInfoList.append({
            "dataset": dataSetName,
            "testFoldIndex": loadedDataset.testFoldIndex,
            "nFolds": loadedDataset.nFolds,
            "testSplitPct": loadedDataset.testSplitPct,
        })
        print(dataSetName + " has class :" +str(np.unique(centralTrainLabel[-1])))
        del loadedDataset
    # Summarise fold info for the combined dataset (all individual datasets share the same split)
    _fi = testFoldInfoList[0] if testFoldInfoList else {}
    testFoldIndex = _fi.get("testFoldIndex", "N/A")
    nFolds        = _fi.get("nFolds",         "N/A")
    testSplitPct  = _fi.get("testSplitPct",   "N/A")

    centralTestLabelAligned = []
    centralTrainLabelAligned = []
    combinedAlignedData = centralTestData
    for index, datasetName in enumerate(datasetList):
        if(datasetName == 'UCI'):
            centralTrainLabelAligned.append(centralTrainLabel[index])
            centralTestLabelAligned.append(centralTestLabel[index])
        elif(datasetName == 'RealWorld'):
            centralTrainLabelAligned.append(np.hstack([REALWORLD_CLIENT[labelIndex] for labelIndex in centralTrainLabel[index]]))
            centralTestLabelAligned.append(np.hstack([REALWORLD_CLIENT[labelIndex] for labelIndex in centralTestLabel[index]]))

        elif(datasetName == 'HHAR'):
            centralTrainLabelAligned.append(np.hstack([HHAR[labelIndex] for labelIndex in centralTrainLabel[index]]))
            centralTestLabelAligned.append(np.hstack([HHAR[labelIndex] for labelIndex in centralTestLabel[index]]))
        elif(datasetName == 'MotionSense'):
            centralTrainLabelAligned.append(np.hstack([MotionSense[labelIndex] for labelIndex in centralTrainLabel[index]]))
            centralTestLabelAligned.append(np.hstack([MotionSense[labelIndex] for labelIndex in centralTestLabel[index]]))
        else:
            centralTrainLabelAligned.append(np.hstack([SHL[labelIndex] for labelIndex in centralTrainLabel[index]]))
            centralTestLabelAligned.append(np.hstack([SHL[labelIndex] for labelIndex in centralTestLabel[index]]))
    centralTrainData = np.vstack((centralTrainData))
    centralTestData = np.vstack((centralTestData))
    centralTrainLabel = np.hstack((centralTrainLabelAligned))
    centralTestLabel = np.hstack((centralTestLabelAligned))
else:
    clientCount = utils.returnClientByDataset(dataSetName)
    datasetLoader = utils.loadDataset(dataSetName,clientCount,dataConfig,randomSeed,mainDir)
    centralTrainData = datasetLoader.centralTrainData
    centralTrainLabel = datasetLoader.centralTrainLabel

    centralTestData = datasetLoader.centralTestData
    centralTestLabel = datasetLoader.centralTestLabel

    clientOrientationTrain = datasetLoader.clientOrientationTrain
    clientOrientationTest = datasetLoader.clientOrientationTest
    orientationsNames = datasetLoader.orientationsNames

    testFoldIndex = datasetLoader.testFoldIndex
    nFolds        = datasetLoader.nFolds
    testSplitPct  = datasetLoader.testSplitPct


# In[ ]:


# If working on RealWorld or HHAR with specified position/device, we remove one and use it as the test set and combine the others for training
if(positionDevice != '' or dataSetName == 'UCI'):
    if(dataSetName == "RealWorld"):
        totalData = np.vstack((centralTrainData,centralTestData))
        totalLabel = np.hstack((centralTrainLabel,centralTestLabel))
        totalOrientation = np.hstack((np.hstack((clientOrientationTrain)), np.hstack((clientOrientationTest))))
        totalIndex = list(range(totalOrientation.shape[0]))
        testDataIndex = np.where(totalOrientation == orientationsNames.index(positionDevice))[0]
        trainDataIndex = np.delete(totalIndex,testDataIndex)

        centralTrainData = totalData[trainDataIndex]
        centralTestData = totalData[testDataIndex]

        centralTrainLabel = totalLabel[trainDataIndex]
        centralTestLabel = totalLabel[testDataIndex]
    elif(dataSetName == "HHAR"):
        totalData = np.vstack((centralTrainData,centralTestData))
        totalLabel = np.hstack((centralTrainLabel,centralTestLabel))
        totalOrientation = np.hstack((np.hstack((clientOrientationTrain)), np.hstack((clientOrientationTest))))
        totalIndex = list(range(totalOrientation.shape[0]))
        # 0 is for nexus
        testDataIndex = np.where(totalOrientation == orientationsNames.index(positionDevice))[0]
        trainDataIndex = np.delete(totalIndex,testDataIndex)

        centralTrainData = totalData[trainDataIndex]
        centralTestData = totalData[testDataIndex]

        centralTrainLabel = totalLabel[trainDataIndex]
        centralTestLabel = totalLabel[testDataIndex]
#         when using positions for evalaution, there is no test set , dev=test is the same
    centralDevData = centralTestData
    centralDevLabel = centralTestLabel
else:
#     using a 70 10 20 ratio
    centralTrainData, centralDevData, centralTrainLabel, centralDevLabel = train_test_split(
        centralTrainData,
        centralTrainLabel,
        test_size=0.125,
        random_state=randomSeed,
        stratify=centralTrainLabel,
    )


# In[ ]:


# Compute class weight
classes = np.unique(centralTrainLabel)
temp_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=centralTrainLabel.ravel(),
)
class_weights = {int(cls): float(weight) for cls, weight in zip(classes, temp_weights)}
for cls in range(activityCount):
    class_weights.setdefault(cls, 1.0)


# In[ ]:


# One Hot of labels
centralTrainLabel = tf.one_hot(centralTrainLabel, activityCount).numpy().astype(np.float32)
centralTestLabel = tf.one_hot(centralTestLabel, activityCount).numpy().astype(np.float32)
centralDevLabel = tf.one_hot(centralDevLabel, activityCount).numpy().astype(np.float32)

centralTrainData = centralTrainData.astype(np.float32)
centralDevData = centralDevData.astype(np.float32)
centralTestData = centralTestData.astype(np.float32)
num_input_channels = int(centralTrainData.shape[-1])
input_shape = (segment_size, num_input_channels)
train_ds = make_dataset(
    centralTrainData,
    centralTrainLabel,
    batch_size=batch_size,
    augment=use_jitter or use_mixup,
    use_jitter_aug=use_jitter,
    jitter_std=jitter_sigma,
    use_mixup_aug=use_mixup,
    mixup_alpha_value=mixup_alpha,
)


# In[ ]:


if use_jitter or use_mixup:
    steps_per_epoch = max(1, centralTrainData.shape[0] // batch_size)
else:
    steps_per_epoch = max(1, math.ceil(centralTrainData.shape[0] / batch_size))
total_steps = steps_per_epoch * localEpoch
warmup_steps = int(total_steps * warmup_ratio)
lr_schedule = WarmUpCosineDecay(
    peak_lr=learningRate,
    total_steps=total_steps,
    warmup_steps=warmup_steps,
    min_lr=min_learning_rate,
)

dataset_checksum_path = os.path.join(mainDir, "datasetStandardized", dataSetName)
dataset_checksum = compute_dataset_checksum(dataset_checksum_path) if os.path.isdir(dataset_checksum_path) else ""

run_metadata = {
    "seed": randomSeed,
    "dataset": dataSetName,
    "data_config": dataConfig,
    "architecture": architecture,
    "architecture_type": architectureType,
    "input_shape": list(input_shape),
    "train_samples": int(centralTrainData.shape[0]),
    "dev_samples": int(centralDevData.shape[0]),
    "test_samples": int(centralTestData.shape[0]),
    "batch_size": int(batch_size),
    "epochs": int(localEpoch),
    "learning_rate_peak": float(learningRate),
    "learning_rate_min": float(min_learning_rate),
    "warmup_ratio": float(warmup_ratio),
    "warmup_steps": int(warmup_steps),
    "total_steps": int(total_steps),
    "weight_decay": float(weight_decay),
    "clipnorm": float(clipnorm),
    "dropout_rate": float(dropout_rate),
    "attention_dropout": float(attention_dropout),
    "mlp_dropout": float(mlp_dropout),
    "token_dropout": float(token_dropout),
    "drop_path_rate": float(drop_path_rate),
    "use_ema": bool(use_ema),
    "ema_decay": float(ema_decay),
    "use_jitter": bool(use_jitter),
    "jitter_sigma": float(jitter_sigma),
    "use_mixup": bool(use_mixup),
    "mixup_alpha": float(mixup_alpha),
    "token_based": bool(tokenBased),
    "enhanced_tokenizer": bool(hartEnhancedTokenizer),
    "activity_count": int(activityCount),
    "device_mode": device_mode,
    "tensorflow_version": tf.__version__,
    "python_version": platform.python_version(),
    "dataset_checksum": dataset_checksum,
    "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "config_path": os.path.abspath(config_path) if config_path else "",
}
save_run_metadata(filepath, run_metadata)

optimizer = build_adamw_optimizer(
    learning_rate=lr_schedule,
    decay=weight_decay,
    grad_clipnorm=clipnorm,
)

if(architecture == "HART"):
    model_classifier = model.HART(
        input_shape,
        activityCount,
        projection_dim=projection_dim,
        patchSize=frameLength,
        timeStep=timeStep,
        filterAttentionHead=filterAttentionHead,
        convKernels=convKernels,
        dropout_rate=dropout_rate,
        attention_dropout=attention_dropout,
        mlp_dropout=mlp_dropout,
        token_dropout=token_dropout,
        drop_path_rate=drop_path_rate,
        useTokens=tokenBased,
        useEnhancedTokenizer=hartEnhancedTokenizer,
    )
else:
    model_classifier = model.mobileHART_XS(input_shape,activityCount)
model_classifier.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)
model_classifier.summary()
model_param_count = int(model_classifier.count_params())
model_size_mb = compute_model_size_mb(model_classifier)
model_flops = compute_model_flops(model_classifier, input_shape)


# In[ ]:


checkpoint_filepath = filepath+"bestValcheckpoint.weights.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=25,
    mode="max",
    restore_best_weights=False,
    verbose=1,
)
ema_callback = EMACallback(decay=ema_decay, monitor="val_accuracy") if use_ema else None
active_callbacks = [checkpoint_callback, early_stop_callback]
if ema_callback is not None:
    active_callbacks.append(ema_callback)

print(f"Train tensors: x={centralTrainData.shape}, y={centralTrainLabel.shape}")
print(f"Validation tensors: x={centralDevData.shape}, y={centralDevLabel.shape}")
assert centralTrainData.ndim == 3 and centralTrainData.shape[1:] == input_shape, f"Expected train x shape (*, {input_shape}), got {centralTrainData.shape}"
assert centralDevData.ndim == 3 and centralDevData.shape[1:] == input_shape, f"Expected dev x shape (*, {input_shape}), got {centralDevData.shape}"
assert centralTestData.ndim == 3 and centralTestData.shape[1:] == input_shape, f"Expected test x shape (*, {input_shape}), got {centralTestData.shape}"
assert model_classifier.input_shape[1:] == input_shape, f"Model input shape mismatch: {model_classifier.input_shape}"
assert model_classifier.output_shape[-1] == activityCount, f"Model output shape mismatch: {model_classifier.output_shape}"

start_time = time.time()
history = model_classifier.fit(
    train_ds,
    validation_data = (centralDevData,centralDevLabel),
    epochs=localEpoch,
    verbose=showTrainVerbose,
    class_weight=None if use_mixup else class_weights,
    callbacks=active_callbacks,
)
end_time = time.time() - start_time

model_classifier.save_weights(filepath + 'bestTrain.weights.h5')
model_classifier.load_weights(checkpoint_filepath)
if ema_callback is not None:
    ema_applied = ema_callback.apply_best_ema()
    if not ema_applied:
        print("EMA weights were not available. Falling back to checkpoint weights.")
print(f"Evaluation tensors: x={centralTestData.shape}, y={centralTestLabel.shape}")
_, accuracy = model_classifier.evaluate(centralTestData, centralTestLabel)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")


# In[ ]:


hkl.dump(history.history,filepath+'history.hkl')


# In[ ]:


def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
    raise ValueError(f"Layer '{layername}' not found in model. Available layers: {[l.name for l in model.layers]}")


# In[ ]:


if(architecture == "HART"):
    finalAccMHAIndex = getLayerIndexByName(model_classifier,"AccMHA_"+str(len(convKernels)-1))
    finalGyroMHAIndex = getLayerIndexByName(model_classifier,"GyroMHA_"+str(len(convKernels)-1))
    finalInputsIndex = getLayerIndexByName(model_classifier,"normalizedInputs_"+str(len(convKernels)-1))
    totalLayer = len(model_classifier.layers)
    classTokenIndex = totalLayer - 4
else:
    finalAccMHAIndex = getLayerIndexByName(model_classifier,"AccMHA_1")
    finalGyroMHAIndex = getLayerIndexByName(model_classifier,"GyroMHA_1")
    finalInputsIndex = getLayerIndexByName(model_classifier,"normalizedInputs_1")
    classTokenIndex = getLayerIndexByName(model_classifier,"GAP")


# In[ ]:


print(f"Prediction tensor: x={centralTestData.shape}")
y_pred = np.argmax(model_classifier.predict(centralTestData), axis=-1)

y_test = np.argmax(centralTestLabel, axis=-1)
weightVal_f1 = f1_score(y_test, y_pred,average='weighted' )
microVal_f1 = f1_score(y_test, y_pred,average='micro')
macroVal_f1 = f1_score(y_test, y_pred,average='macro')

# ── Early results print (before visualisations / TFLite export) ───────────
_split_tag = f"  [fold {testFoldIndex} of {nFolds} — {testSplitPct}% test split]" \
    if testFoldIndex not in (None, "N/A") else ""
print("\n" + "=" * 54)
print(" Test Results")
print("=" * 54)
print(f"  Accuracy:    {round(accuracy * 100, 2):.2f}%{_split_tag}")
print(f"  Weighted F1: {round(weightVal_f1 * 100, 2):.2f}%{_split_tag}")
print(f"  Micro F1:    {round(microVal_f1 * 100, 2):.2f}%{_split_tag}")
print(f"  Macro F1:    {round(macroVal_f1 * 100, 2):.2f}%{_split_tag}")
print(f"  Train acc (peak):  {round(np.max(history.history['accuracy']) * 100, 2):.2f}%")
print(f"  Val acc   (peak):  {round(np.max(history.history['val_accuracy']) * 100, 2):.2f}%")
print(f"  Model size:        {round(model_size_mb, 2)} MB  ({model_param_count:,} params)")
print("=" * 54 + "\n")
# ─────────────────────────────────────────────────────────────────────────

modelStatistics = {
"Results on server model on ALL testsets" : '',
"\nTrain:" : utils.roundNumber(np.max(history.history['accuracy'])),
"\nValidation:" : utils.roundNumber(np.max(history.history['val_accuracy'])),
"\nTest weighted f1:" : utils.roundNumber(weightVal_f1),
"\nTest micro f1:": utils.roundNumber(microVal_f1),
"\nTest macro f1:": utils.roundNumber(macroVal_f1),
"\nTest split (fold index):" : testFoldIndex,
"\nTest split (total folds):" : nFolds,
"\nTest split size (%):" : testSplitPct,
"\nModel params:": model_param_count,
"\nModel size (MB):": round(model_size_mb, 4),
"\nSize (MB):": round(model_size_mb, 4),
"\nFLOPs (batch=1):": model_flops if model_flops is not None else "N/A",
"\nFLOPs operations (batch=1):": model_flops if model_flops is not None else "N/A",
"\nComplexity matrix:": "ComplexityMatrix.csv",
}    
with open(filepath +'GlobalACC.csv','w') as f:
    w = csv.writer(f)
    w.writerows(modelStatistics.items())

complexity_matrix = pd.DataFrame(
    [
        {"metric": "params",      "value": model_param_count},
        {"metric": "size_mb",     "value": round(model_size_mb, 4)},
        {"metric": "flops_batch1","value": model_flops if model_flops is not None else "N/A"},
    ]
)
complexity_matrix.to_csv(filepath + "ComplexityMatrix.csv", index=False)

# ── Complexity Matrix chart ───────────────────────────────────────────────
_cm_metrics = ["Parameters", "Size (MB)", "FLOPs\n(batch=1)"]
_cm_values  = [
    model_param_count,
    round(model_size_mb, 4),
    model_flops if model_flops is not None else 0,
]
_cm_labels  = [
    f"{model_param_count:,}",
    f"{round(model_size_mb, 4)} MB",
    f"{model_flops:,}" if model_flops is not None else "N/A",
]
_cm_colors  = ["#4C72B0", "#55A868", "#C44E52"]

fig_cm, axes_cm = plt.subplots(1, 3, figsize=(11, 4))
fig_cm.suptitle(
    f"Model Complexity Matrix  —  {architecture} / {dataSetName}",
    fontsize=13, fontweight="bold", y=1.02,
)
for ax, metric, value, label, color in zip(
    axes_cm, _cm_metrics, _cm_values, _cm_labels, _cm_colors
):
    bar = ax.bar([metric], [value], color=color, alpha=0.85, width=0.45)
    ax.set_title(metric, fontsize=11)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:.4f}")
    )
    ax.bar_label(bar, labels=[label], padding=6, fontsize=10, fontweight="bold")
    ax.set_ylim(0, value * 1.25 if value > 0 else 1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

fig_cm.tight_layout()
complexity_matrix_png = filepath + "ComplexityMatrix.png"
fig_cm.savefig(complexity_matrix_png, bbox_inches="tight", dpi=150)
if "agg" not in plt.get_backend().lower():
    plt.show()
plt.close(fig_cm)
print(f"Complexity matrix chart saved: {complexity_matrix_png}")
# ─────────────────────────────────────────────────────────────────────────

global_metrics_path = filepath + 'GlobalACC.csv'
complexity_matrix_path = filepath + "ComplexityMatrix.csv"
confusion_matrix_path = filepath + 'HeatMap.png'
tflite_output_path = filepath + architecture + '_int8.tflite'


# In[ ]:


xLabel = np.arange(0,2.58,0.02)
idx = np.round(np.linspace(0, len(xLabel) - 1, 8)).astype(int)
s = pd.Series(y_test)
indices = [v.values[min(2, len(v.values) - 1)] for k,v in s.groupby(s).groups.items()]
segmentTime = [x for x in range(0,segment_size - frameLength + timeStep,timeStep)]
inputModel = tf.keras.Model(inputs=model_classifier.inputs, outputs=model_classifier.layers[finalInputsIndex].output)

for index, classLoc in enumerate(indices):
    inputsToAttention = inputModel(np.expand_dims(centralTestData[classLoc], 0))
    _,attentionAccWeights = model_classifier.layers[finalAccMHAIndex](inputsToAttention, return_attention_scores=True)
    _,attentionGyroWeights = model_classifier.layers[finalGyroMHAIndex](inputsToAttention, return_attention_scores=True)
    if(tokenBased):
        attentionScores = np.mean(attentionAccWeights[0],axis = 0)[0,1:]
    else:
         attentionScores = np.mean(attentionAccWeights[0],axis = 0)[0,:]
    attentionScoresNorm = ((attentionScores - min(attentionScores))/(max(attentionScores) - min(attentionScores)) -1) * - 0.5
    channel_count = centralTestData.shape[2]
    has_gyro = channel_count >= 6
    gs = gridspec.GridSpec(2,1) if has_gyro else gridspec.GridSpec(1,1)
    fig = plt.figure()
    plt.title("Attention Map for "+ACTIVITY_LABEL[index]+" ",size =16)    
    plt.margins(x=0)
    plt.tick_params(
    axis='both',        
    which='both',      
    labelleft = False,
    left = False,
    bottom=False,      
    top=False,         
    labelbottom=False) 

    
    ax = fig.add_subplot(gs[0])
    ax.margins(x=0)

    ax.plot( centralTestData[classLoc][:,0], label = "x-axis")
    ax.plot( centralTestData[classLoc][:,1], label = "y-axis")
    ax.plot( centralTestData[classLoc][:,2], label = "z-axis")
    for barIndex, starTime in enumerate(segmentTime):
        ax.axvspan(starTime, starTime + frameLength, facecolor='black', alpha=float(attentionScoresNorm[barIndex]),zorder=4)
        
    ax.set_ylabel(r'Acc ($m/s^2$)', size =16)
    ax.get_yaxis().set_label_coords(-0.1,0.5)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        labelbottom=False) 
    plt.legend(loc='upper right', framealpha = 0.7)

    if has_gyro:
        if(tokenBased):
            attentionScores = np.mean(attentionGyroWeights[0],axis = 0)[0,1:]
        else:
            attentionScores = np.mean(attentionGyroWeights[0],axis = 0)[0,:]
        attentionScoresNorm = ((attentionScores - min(attentionScores))/(max(attentionScores) - min(attentionScores)) -1) * - 0.5
        
        ax = fig.add_subplot(gs[1], sharex=ax)
        ax.margins(x=0)

        ax.plot( centralTestData[classLoc][:,3], label = "x-axis")
        ax.plot( centralTestData[classLoc][:,4],label = "y-axis")
        ax.plot( centralTestData[classLoc][:,5], label = "z-axis")
        
        for barIndex, starTime in enumerate(segmentTime):
            ax.axvspan(starTime, starTime + frameLength, facecolor='black', alpha=float(attentionScoresNorm[barIndex]),zorder=99)

        ax.set_ylabel(r'Gyro (rad/s)', size =16)
        ax.get_yaxis().set_label_coords(-0.1,0.5)
        ax.set_xticks([0,32,64,96,128])
        ax.set_xticklabels([0,0.64,1.28,1.9, 2.56])
        ax.set_xlabel("Time (s)", size =16)
        ax.margins(x=0)
    else:
        ax.set_xticks([0,32,64,96,128])
        ax.set_xticklabels([0,0.64,1.28,1.9, 2.56])
        ax.set_xlabel("Time (s)", size =16)
        

    plt.savefig(attentionPath+ACTIVITY_LABEL[index]+"MeanHeadAttention.png", bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.clf()


# In[ ]:


utils.plot_learningCurve(history,localEpoch,filepath) 


# In[ ]:


totalLayer = len(model_classifier.layers)
classTokenIndex = totalLayer - 4
intermediateModel = utils.extract_intermediate_model_from_base_model(model_classifier,classTokenIndex)


# In[ ]:


perplexity = 30.0
embeddings = intermediateModel.predict(centralTestData, batch_size=batch_size)
del intermediateModel
tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=showTrainVerbose, random_state=randomSeed)
tsne_projections = tsne_model.fit_transform(embeddings)
labels_argmax = np.argmax(centralTestLabel, axis=-1)
unique_labels = np.unique(labels_argmax)


# In[ ]:


if((dataSetName == 'RealWorld' or dataSetName == 'HHAR') and positionDevice == ''):
    utils.projectTSNEWithPosition(dataSetName,architecture+"_TSNE_Embeds",filepath,ACTIVITY_LABEL,labels_argmax,orientationsNames,clientOrientationTest,tsne_projections,unique_labels)
else:
    utils.projectTSNE(architecture+"_TSNE_Embeds",filepath,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels)


# In[ ]:


all_class_ids = np.arange(activityCount)
results = confusion_matrix(y_test, y_pred, labels=all_class_ids)
display_labels = [ACTIVITY_LABEL[i] for i in all_class_ids]
df_cm = pd.DataFrame(results, index=display_labels, columns=display_labels)
plt.figure(figsize = (14,14))
sns.set(font_scale=1.4) 
sns.heatmap(df_cm, annot=True,cmap=plt.cm.Blues,cbar=False)
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.savefig(filepath+'HeatMap.png')


# In[ ]:


export_tflite_int8(
    model_classifier,
    centralTestData[:500],
    tflite_output_path,
)


# In[ ]:


print_final_test_summary(
    test_accuracy=accuracy,
    weighted_f1=weightVal_f1,
    micro_f1=microVal_f1,
    macro_f1=macroVal_f1,
    confusion_matrix_df=df_cm,
    global_metrics_path=global_metrics_path,
    confusion_matrix_path=confusion_matrix_path,
    complexity_matrix_path=complexity_matrix_path,
    tflite_path=tflite_output_path,
    test_fold_index=testFoldIndex,
    n_folds=nFolds,
    test_split_pct=testSplitPct,
    y_true=y_test,
    y_pred=y_pred,
    activity_labels=ACTIVITY_LABEL,
    n_test_samples=len(y_test),
)
print("Training Done!")


# In[ ]:




