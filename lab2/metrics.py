from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score
from sklearn.metrics import auc
from sklearn.datasets import make_classification
import numpy as np


def accuracy(y_true, y_pred):
    # TP, TN, FP, FN
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


def precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    return precision


def recall(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    return recall


def auc_roc(y_true, y_pred_prob):
    thresholds = np.linspace(0, 1, num=1000)

    tpr = []
    fpr = []

    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        tpr.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    auc = np.trapezoid(tpr[::-1], fpr[::-1], dx=0.1)
    return auc


def auc_pr(y_true, y_pred_prob):
    thresholds = np.linspace(0, 1, num=10000)
    precision_vals = []
    recall_vals = []

    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        precision_vals.append(precision)
        recall_vals.append(recall)

    precision_vals = np.array(precision_vals)
    recall_vals = np.array(recall_vals)

    auc = np.trapezoid(precision_vals[::-1], recall_vals[::-1], dx=0.01)
    return auc


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    f1 = 2 * (p * r) / (p + r) if (p + r) != 0 else 0
    return f1

