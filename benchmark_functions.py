import sys
import os
from pathlib import Path

sys.path.insert(0, "../")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
)
from matplotlib import pyplot as plt
from IPython.display import Image, display
from scipy import stats

# cleanlab imports
import cleanlab
from cleanlab.multiannotator import (
    get_label_quality_multiannotator,
    get_multiannotator_stats,
)
from cleanlab.rank import get_label_quality_scores, get_label_quality_ensemble_scores
from cleanlab.internal.label_quality_utils import get_normalized_entropy
from cleanlab.filter import find_label_issues

# local imports
from utils.eval_metrics import lift_at_k
from utils.active_learning_scores import least_confidence

# experimental version of label quality ensemble scores with additional weighting schemes
from utils.label_quality_ensemble_scores_experimental import (
    get_label_quality_ensemble_scores_experimental,
)

import warnings

warnings.filterwarnings("ignore")

path = os.getcwd()
fig_size_small = (5, 8)
fig_size_big = (15, 10)


def get_annotator_accuracy(c10h_labels, c10h_true_labels):
    annotator_accuracy = pd.DataFrame(c10h_labels).apply(
        lambda s: np.mean(s[pd.notna(s)] == c10h_true_labels[pd.notna(s)])
    )
    df_describe = pd.DataFrame(annotator_accuracy, columns=["score"])
    return df_describe


def get_consensus_label_accuracy(consensus_labels, true_labels):
    consensus_labels_accuracy = (true_labels == consensus_labels).sum() / 10000
    # print('Consensus label accuracy: ', consensus_labels_accuracy)
    # print('\nPer class scores:')
    p, r, f1, _ = precision_recall_fscore_support(true_labels, consensus_labels)
    results_df = pd.DataFrame(zip(p, r, f1), columns=["precision", "recall", "f1"])
    return consensus_labels_accuracy, results_df


def get_spearman_correlation(x, y):
    num_nans_x = np.sum(np.isnan(x))
    num_nans_y = np.sum(np.isnan(y))

    if num_nans_x > 0:
        x = np.nan_to_num(x)
        print("First param contains nans. Replacing", num_nans_x, "nans with 0")
    if num_nans_y > 0:
        y = np.nan_to_num(y)
        print("First param contains nans. Replacing", num_nans_y, "nans with 0")

    return stats.spearmanr(x, y)


def benchmark_results(c10h_labels, c10h_true_labels, pred_probs, methods, model):
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

        # compute accuracy of detecting label errors
        auroc = roc_auc_score(label_errors_target, -ranked_quality)

        consensus_labels = label_quality_multiannotator["consensus_label"]
        consensus_labels_accuracy, results_df = get_consensus_label_accuracy(
            consensus_labels, c10h_true_labels
        )

        quality_score = multiannotator_stats.sort_index()["overall_quality"].values

        annotator_accuracy_df = get_annotator_accuracy(c10h_labels, c10h_true_labels)
        accuracy = annotator_accuracy_df["score"].values

        speaman_corr = get_spearman_correlation(quality_score, accuracy)

        # save results
        results = {
            "dataset": "cifar10",
            "model": model,
            "consensus_method": consensus_method,
            "quality_method": quality_method,
            "auroc": auroc,
            "consensus_labels_accuracy": consensus_labels_accuracy,
            "spearman_correlation": speaman_corr.correlation,
        }

        # save results
        results_list.append(results)

        # compute precision-recall curve using label quality scores
        precision, recall, thresholds = precision_recall_curve(
            label_errors_target, -ranked_quality
        )

        # compute au-roc curve using label quality scores
        fpr, tpr, thresholds = roc_curve(label_errors_target, -ranked_quality)

        # precision_recall_curve_results = {
        #     "dataset": "cifar10",
        #     "model": model,
        #     "consensus_method": consensus_method,
        #     "quality_method": quality_method,
        #     "label_quality_scores": label_quality_scores,
        #     "precision": precision,
        #     "recall": recall,
        #     "thresholds": thresholds,
        # }

        # plot prc
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, label=f"{consensus_method}-{quality_method}")
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.title(
            "Precision-Recall Curve \n Model: resnet-18", fontsize=14, fontweight="bold"
        )
        plt.legend()

        # plot roc
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f"{consensus_method}-{quality_method}")
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("AU ROC Curve \n Model: resnet-18", fontsize=14, fontweight="bold")
        plt.legend()

    return results_list
