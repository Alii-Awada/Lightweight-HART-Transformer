#!/usr/bin/env python
# coding: utf-8

from collections import Counter

import numpy as np


# WHO intensity thresholds (METs)
# Light < 3.0  |  Moderate 3.0-5.99  |  Vigorous >= 6.0
ACTIVITY_MET = {
    "lying": 1.0,
    "sitting": 1.3,
    "standing": 1.5,
    "walking": 3.5,
    "running": 8.0,
    "cycling": 6.0,
    "nordic_walking": 4.8,
    "ascending_stairs": 4.0,
    "descending_stairs": 3.5,
}


def met_to_intensity(met):
    if met < 3.0:
        return "light"
    if met < 6.0:
        return "moderate"
    return "vigorous"


def predict_intensity(activity_probs, activity_labels):
    """Map activity softmax outputs to WHO intensity bands and METs."""
    pred_idx = np.argmax(activity_probs, axis=1)
    pred_names = [activity_labels[i] for i in pred_idx]
    pred_mets = np.array([ACTIVITY_MET[name] for name in pred_names], dtype=np.float32)
    intensities = [met_to_intensity(met) for met in pred_mets]
    return intensities, pred_mets


def who_weekly_minutes(intensity_list, window_step_sec=0.64):
    """Accumulate WHO moderate-equivalent minutes from windowed predictions."""
    counts = Counter(intensity_list)
    minutes = {
        key: counts[key] * window_step_sec / 60.0
        for key in ["light", "moderate", "vigorous"]
    }
    met_minutes = minutes["moderate"] + (minutes["vigorous"] * 2.0)
    return minutes, met_minutes
