import sys
import os
from pathlib import Path

sys.path.insert(0, "../")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    precision_recall_fscore_support,
    auc,
)
from matplotlib import pyplot as plt
from scipy import stats

# cleanlab imports
from cleanlab.multiannotator import get_label_quality_multiannotator

import warnings

warnings.filterwarnings("ignore")

path = os.getcwd()
fig_size_small = (5, 8)
fig_size_big = (10, 7)


def get_annotator_accuracy(annotator_labels, true_labels):
    annotator_accuracy = pd.DataFrame(annotator_labels).apply(
        lambda s: np.mean(s[pd.notna(s)] == true_labels[pd.notna(s)])
    )
    df_describe = pd.DataFrame(annotator_accuracy, columns=["score"])
    return df_describe


def get_consensus_label_accuracy(consensus_labels, true_labels):
    consensus_labels_accuracy = np.mean(true_labels == consensus_labels)
    p, r, f1, _ = precision_recall_fscore_support(true_labels, consensus_labels)
    results_df = pd.DataFrame(zip(p, r, f1), columns=["precision", "recall", "f1"])
    return consensus_labels_accuracy, results_df


def get_spearman_correlation(x, y):
    num_nans_x = np.sum(np.isnan(x))
    num_nans_y = np.sum(np.isnan(y))

    if num_nans_x > 0:
        x = np.nan_to_num(x)
    if num_nans_y > 0:
        y = np.nan_to_num(y)

    return stats.spearmanr(x, y)


def lift_at_k(y_true, y_score, k=100):
    """Compute Lift at K evaluation metric"""
    sort_indices = np.argsort(y_score)
    # compute lift for the top k values
    lift_at_k = y_true[sort_indices][-k:].mean() / y_true.mean()

    return lift_at_k


def benchmark_results(
    c10h_labels,
    c10h_true_labels,
    pred_probs,
    methods,
    model_name,
    dataset_name,
    add_model=False,
):
    plt.rcParams["figure.figsize"] = fig_size_big
    results_list = []

    c10h_labels = pd.DataFrame(c10h_labels)

    for consensus_method, quality_method in methods:
        (
            label_quality_multiannotator,
            multiannotator_stats,
        ) = get_label_quality_multiannotator(
            c10h_labels,
            pred_probs,
            consensus_method=consensus_method,
            quality_method=quality_method,
            return_annotator_stats=True,
            verbose=False,
        )

        # create boolean mask of label errors
        labels = label_quality_multiannotator["consensus_label"]
        label_errors_target = (
            labels != c10h_true_labels
        )  # labels can change to annotator labels!!

        # compute scores
        label_quality_scores = label_quality_multiannotator["quality_of_consensus"]
        ranked_quality = label_quality_multiannotator["ranked_quality"]

        consensus_labels = label_quality_multiannotator["consensus_label"]
        consensus_labels_accuracy, results_df = get_consensus_label_accuracy(
            consensus_labels, c10h_true_labels
        )

        quality_score = multiannotator_stats.sort_index()["overall_quality"].values

        annotator_accuracy_df = get_annotator_accuracy(c10h_labels, c10h_true_labels)
        accuracy = annotator_accuracy_df["score"].values

        speaman_corr = get_spearman_correlation(quality_score, accuracy)

        # compute precision-recall curve using label quality scores
        precision, recall, thresholds = precision_recall_curve(
            label_errors_target, -ranked_quality
        )

        # compute au-roc curve using label quality scores
        fpr, tpr, thresholds = roc_curve(label_errors_target, -ranked_quality)

        # compute accuracy of detecting label errors
        auroc = roc_auc_score(label_errors_target, -ranked_quality)
        auprc = auc(recall, precision)

        lift_at_k_dict = {}
        for k in [10, 50, 100, 300, 500, 1000]:
            lift_at_k_dict[f"lift_at_{k}"] = lift_at_k(
                label_errors_target, -ranked_quality, k=k
            )

        if add_model == True:
            consensus_method += "_with_model"
            quality_method += "_with_model"

        # save results
        results = {
            "dataset": dataset_name,
            "model": model_name,
            "consensus_method": consensus_method,
            "quality_method": quality_method,
            "consensus_quality_auroc": auroc,
            "consensus_quality_auprc": auprc,
            "consensus_labels_accuracy": consensus_labels_accuracy,
            "annotator_quality_spearman_corr": speaman_corr.correlation,
        }

        results.update(lift_at_k_dict)

        # save results
        results_list.append(results)

        # plot prc
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, label=f"{quality_method}")
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.title(
            f"Precision-Recall Curve \n Model: {model_name} \n Dataset: {dataset_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend()

        # plot roc
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f"{quality_method}")
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title(
            f"AU ROC Curve \n Model: {model_name} \n Dataset: {dataset_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend()

    return results_list
