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
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

SENSOR_BASE_COLUMN = {
    "hand": 3,
    "chest": 20,
    "ankle": 37,
}

SENSOR_LOCATION = "hand"
SEGMENT_SIZE = 128
STEP_SIZE = 64


def get_usecols(sensor_location):
    base = SENSOR_BASE_COLUMN[sensor_location]
    acc16 = [base + 1, base + 2, base + 3]
    return [1] + acc16


def load_subject(filepath, sensor_location):
    usecols = get_usecols(sensor_location)
    df = pd.read_csv(filepath, header=None, sep=r"\s+", usecols=usecols, engine="python")
    df = df.dropna()
    df = df[df[1].isin(ACTIVITY_MAP.keys())]
    labels = df[1].astype(int).to_numpy()
    data = df.drop(columns=[1]).to_numpy(dtype=np.float32)
    return data, labels


def create_windows(data, labels, segment_size, step_size):
    segments = []
    segment_labels = []
    start = 0
    while start + segment_size <= len(data):
        end = start + segment_size
        window_labels = labels[start:end]
        if np.all(window_labels == window_labels[0]):
            segments.append(data[start:end])
            segment_labels.append(window_labels[0])
        start += step_size
    if len(segments) == 0:
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
        data, labels = load_subject(subject_path, SENSOR_LOCATION)
        if data.size == 0:
            continue
        windows, window_labels = create_windows(
            data, labels, SEGMENT_SIZE, STEP_SIZE
        )
        if windows.size == 0:
            continue
        mapped_labels = np.asarray([id_to_index[x] for x in window_labels], dtype=np.int32)
        all_subject_data.append(windows)
        all_subject_labels.append(mapped_labels)

    if len(all_subject_data) == 0:
        raise RuntimeError("No PAMAP2 segments created. Check filters and window size.")

    combined_data = np.vstack(all_subject_data)
    acc = combined_data[:, :, :3]
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)

    normalized_subject_data = []
    for subject_data in all_subject_data:
        subject_acc = (subject_data[:, :, :3] - acc_mean) / acc_std
        normalized_subject_data.append(subject_acc)

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

    print("PAMAP2 processing finished")


if __name__ == "__main__":
    main()
