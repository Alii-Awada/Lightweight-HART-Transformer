#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import hickle as hkl


ACTIVITY_MAP = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
}

SENSOR_BASE_COLUMN = {
    "hand": 3,
    "chest": 20,
    "ankle": 37,
}

SENSOR_LOCATION = "hand"
SENSOR_CHANNEL_MODE = "acc_gyro"  # acc_only | acc_gyro
SEGMENT_SIZE = 128
STEP_SIZE = 64


def get_usecols(sensor_location):
    base = SENSOR_BASE_COLUMN[sensor_location]
    acc16 = [base + 1, base + 2, base + 3]
    if SENSOR_CHANNEL_MODE == "acc_only":
        return [0, 1] + acc16
    if SENSOR_CHANNEL_MODE == "acc_gyro":
        gyro = [base + 7, base + 8, base + 9]
        return [0, 1] + acc16 + gyro
    raise ValueError("Unsupported SENSOR_CHANNEL_MODE: " + str(SENSOR_CHANNEL_MODE))


def load_subject(filepath, sensor_location):
    usecols = get_usecols(sensor_location)
    df = pd.read_csv(filepath, header=None, sep=r"\s+", usecols=usecols, engine="python")
    df = df.dropna()
    df = df[df[1].isin(ACTIVITY_MAP.keys())]
    timestamps = df[0].to_numpy(dtype=np.float32)
    labels = df[1].astype(int).to_numpy()
    data = df.drop(columns=[0, 1]).to_numpy(dtype=np.float32)
    return data, labels, timestamps


def create_windows_gapaware(data, labels, timestamps, segment_size, step_size, max_gap_ms=25):
    """Window each contiguous chunk independently."""
    dt = np.diff(timestamps)
    gap_indices = np.where(dt > max_gap_ms)[0] + 1
    chunk_boundaries = [0] + gap_indices.tolist() + [len(data)]

    segments = []
    segment_labels = []
    for start_idx, end_idx in zip(chunk_boundaries, chunk_boundaries[1:]):
        chunk_data = data[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        pos = 0
        while pos + segment_size <= len(chunk_data):
            window_labels = chunk_labels[pos:pos + segment_size]
            if np.all(window_labels == window_labels[0]):
                segments.append(chunk_data[pos:pos + segment_size])
                segment_labels.append(window_labels[0])
            pos += step_size
    if not segments:
        return (
            np.empty((0, segment_size, data.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return np.stack(segments).astype(np.float32), np.asarray(segment_labels, dtype=np.int32)


def main():
    protocol_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "PAMAP2_Dataset", "Protocol")
    )
    if not os.path.isdir(protocol_dir):
        raise FileNotFoundError(
            "PAMAP2 Protocol folder not found at: " + protocol_dir
        )

    activity_ids = sorted(ACTIVITY_MAP.keys())
    id_to_index = {activity_id: idx for idx, activity_id in enumerate(activity_ids)}

    subject_files = sorted(
        [
            os.path.join(protocol_dir, name)
            for name in os.listdir(protocol_dir)
            if name.endswith(".dat")
        ]
    )
    if len(subject_files) == 0:
        raise FileNotFoundError("No .dat files found in: " + protocol_dir)

    all_subject_data = []
    all_subject_labels = []

    for subject_path in subject_files:
        data, labels, timestamps = load_subject(subject_path, SENSOR_LOCATION)
        if data.size == 0:
            continue
        windows, window_labels = create_windows_gapaware(
            data, labels, timestamps, SEGMENT_SIZE, STEP_SIZE
        )
        if windows.size == 0:
            continue
        mapped_labels = np.asarray([id_to_index[x] for x in window_labels], dtype=np.int32)
        all_subject_data.append(windows)
        all_subject_labels.append(mapped_labels)

    if len(all_subject_data) == 0:
        raise RuntimeError("No PAMAP2 segments created. Check filters and window size.")

    if len(all_subject_data) < 2:
        raise RuntimeError("Train-only normalization requires at least two PAMAP2 subjects.")

    normalized_subject_data = []
    for held_out_subject, subject_data in enumerate(all_subject_data):
        train_subject_indices = [
            i for i in range(len(all_subject_data)) if i != held_out_subject
        ]
        combined_train = np.vstack(
            [all_subject_data[i] for i in train_subject_indices]
        )
        channel_mean = np.mean(combined_train, axis=(0, 1), keepdims=True)
        channel_std = np.std(combined_train, axis=(0, 1), keepdims=True)
        channel_std = np.where(channel_std < 1e-8, 1.0, channel_std)
        normalized_subject_data.append((subject_data - channel_mean) / channel_std)

    data_name = "PAMAP2"
    os.makedirs("datasetStandardized/" + data_name, exist_ok=True)
    for idx, (subject_data, subject_label) in enumerate(
        zip(normalized_subject_data, all_subject_labels)
    ):
        hkl.dump(
            subject_data,
            "datasetStandardized/" + data_name + "/UserData" + str(idx) + ".hkl",
        )
        hkl.dump(
            subject_label,
            "datasetStandardized/" + data_name + "/UserLabel" + str(idx) + ".hkl",
        )

    with open("datasetStandardized/" + data_name + "/activity_labels.txt", "w") as f:
        for activity_id in activity_ids:
            f.write(
                str(id_to_index[activity_id])
                + ","
                + str(activity_id)
                + ","
                + ACTIVITY_MAP[activity_id]
                + "\n"
            )

    print(
        "PAMAP2 processing finished "
        f"(sensor={SENSOR_LOCATION}, mode={SENSOR_CHANNEL_MODE}, channels={normalized_subject_data[0].shape[-1]})"
    )


if __name__ == "__main__":
    main()
