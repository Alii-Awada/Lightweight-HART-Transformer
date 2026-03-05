#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import hickle as hkl
import seaborn as sns
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


class dataHolder:
    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []
    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    clientOrientationTrain = []
    clientOrientationTest = []
    orientationsNames = None
    activityLabels = []
    clientCount = None


def returnClientByDataset(dataSetName):
    if dataSetName == "PAMAP2":
        return 9
    raise ValueError("Unknown dataset")


def projectTSNE(fileName, filepath, ACTIVITY_LABEL, labels_argmax, tsne_projections, unique_labels):
    plt.figure(figsize=(16, 16))
    graph = sns.scatterplot(
        x=tsne_projections[:, 0],
        y=tsne_projections[:, 1],
        hue=labels_argmax,
        palette=sns.color_palette(n_colors=len(unique_labels)),
        s=50,
        alpha=1.0,
        rasterized=True,
    )
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(ACTIVITY_LABEL[int(label)])

    plt.tick_params(axis="both", which="both", bottom=False, top=False, labelleft=False, labelbottom=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath + fileName + ".svg", bbox_inches="tight", format="svg")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def projectTSNEWithPosition(
    dataSetName,
    fileName,
    filepath,
    ACTIVITY_LABEL,
    labels_argmax,
    orientationsNames,
    clientOrientationTest,
    tsne_projections,
    unique_labels,
):
    classData = [ACTIVITY_LABEL[i] for i in labels_argmax]
    orientationData = [orientationsNames[i] for i in np.hstack((clientOrientationTest))]
    orientationName = "Position" if dataSetName == "RealWorld" else "Device"
    pandaData = {
        "col1": tsne_projections[:, 0],
        "col2": tsne_projections[:, 1],
        "Classes": classData,
        orientationName: orientationData,
    }
    pandaDataFrame = pd.DataFrame(data=pandaData)

    plt.figure(figsize=(16, 16))
    sns.scatterplot(
        data=pandaDataFrame,
        x="col1",
        y="col2",
        hue="Classes",
        style=orientationName,
        palette=sns.color_palette(n_colors=len(unique_labels)),
        s=50,
        alpha=1.0,
        rasterized=True,
    )
    plt.tick_params(axis="both", which="both", bottom=False, top=False, labelleft=False, labelbottom=False)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath + fileName + ".png", bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close()


def loadDataset(dataSetName, clientCount, dataConfig, randomSeed, mainDir, StratifiedSplit=True):
    if dataSetName != "PAMAP2":
        raise ValueError("Unknown dataset")

    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []

    clientData = []
    clientLabel = []

    for i in range(0, clientCount):
        clientData.append(hkl.load(mainDir + "datasetStandardized/PAMAP2/UserData" + str(i) + ".hkl"))
        clientLabel.append(hkl.load(mainDir + "datasetStandardized/PAMAP2/UserLabel" + str(i) + ".hkl"))

    for i in range(0, clientCount):
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        skf.get_n_splits(clientData[i], clientLabel[i])
        trainIndex = []
        testIndex = []
        for enu_index, (train_index, test_index) in enumerate(skf.split(clientData[i], clientLabel[i])):
            if enu_index != 2:
                trainIndex.append(test_index)
            else:
                testIndex = test_index
        trainIndex = np.hstack((trainIndex))
        clientDataTrain.append(clientData[i][trainIndex])
        clientLabelTrain.append(clientLabel[i][trainIndex])
        clientDataTest.append(clientData[i][testIndex])
        clientLabelTest.append(clientLabel[i][testIndex])

    centralTrainData = np.vstack((clientDataTrain))
    centralTrainLabel = np.hstack((clientLabelTrain))

    centralTestData = np.vstack((clientDataTest))
    centralTestLabel = np.hstack((clientLabelTest))

    dataReturn = dataHolder
    dataReturn.clientDataTrain = clientDataTrain
    dataReturn.clientLabelTrain = clientLabelTrain
    dataReturn.clientDataTest = clientDataTest
    dataReturn.clientLabelTest = clientLabelTest
    dataReturn.centralTrainData = centralTrainData
    dataReturn.centralTrainLabel = centralTrainLabel
    dataReturn.centralTestData = centralTestData
    dataReturn.centralTestLabel = centralTestLabel
    dataReturn.clientOrientationTrain = []
    dataReturn.clientOrientationTest = []
    dataReturn.orientationsNames = None
    return dataReturn


def plot_learningCurve(history, epochs, filepath):
    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
    epoch_count = len(history.history[acc_key])
    epoch_range = range(1, epoch_count + 1)

    plt.plot(epoch_range, history.history[acc_key])
    plt.plot(epoch_range, history.history[val_acc_key])
    plt.plot(epoch_range, history.history[val_acc_key], markevery=[np.argmax(history.history[val_acc_key])], ls="", marker="o", color="orange")
    plt.plot(epoch_range, history.history[acc_key], markevery=[np.argmax(history.history[acc_key])], ls="", marker="o", color="blue")

    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="lower right")
    plt.savefig(filepath + "LearningAccuracy.svg", bbox_inches="tight", format="svg")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.clf()

    plt.plot(epoch_range, history.history["loss"])
    plt.plot(epoch_range, history.history["val_loss"])
    plt.plot(epoch_range, history.history["loss"], markevery=[np.argmin(history.history["loss"])], ls="", marker="o", color="blue")
    plt.plot(epoch_range, history.history["val_loss"], markevery=[np.argmin(history.history["val_loss"])], ls="", marker="o", color="orange")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper right")
    plt.savefig(filepath + "ModelLoss.svg", bbox_inches="tight", format="svg")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.clf()


def roundNumber(toRoundNb):
    return round(toRoundNb, 4) * 100


def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[intermediate_layer].output,
        name=base_model.name + "_layer_" + str(intermediate_layer),
    )
    return model
