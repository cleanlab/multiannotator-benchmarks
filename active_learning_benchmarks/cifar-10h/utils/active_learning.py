import numpy as np
import pandas as pd
from cleanlab.internal.label_quality_utils import get_normalized_entropy
from cleanlab.internal.util import get_num_classes, value_counts


def handle_missing_classes(
    multiannotator_labels, majority_vote_label, pred_probs, unlabeled=False
):
    unique_ma_labels = np.unique(multiannotator_labels)
    unique_ma_labels = unique_ma_labels[~np.isnan(unique_ma_labels)]
    missing_set = set(unique_ma_labels) - set(np.unique(majority_vote_label))

    if len(missing_set) > 0:
        overall_label_counts = multiannotator_labels.apply(
            lambda s: np.bincount(s.dropna(), minlength=len(unique_ma_labels)),
            axis=1,
        ).sum(axis=0)

        normalized_overall_label_counts = overall_label_counts / np.sum(
            overall_label_counts
        )

        if unlabeled:
            missing_label_pred_probs = np.full(
                (len(pred_probs), len(unique_ma_labels)),
                normalized_overall_label_counts,
            )

        else:
            empirical_label_distribution = np.vstack(
                multiannotator_labels.apply(
                    lambda s: np.bincount(s.dropna(), minlength=len(unique_ma_labels))
                    / len(s.dropna()),
                    axis=1,
                )
            )

            missing_label_pred_probs = (
                normalized_overall_label_counts + empirical_label_distribution
            )

    for missing_class in missing_set:
        missing_class = int(missing_class)
        print(f"Class {missing_class} is missing, adding miniscule pred_probs")

        pred_probs = np.concatenate(
            (
                pred_probs[:, :missing_class],
                missing_label_pred_probs[:, missing_class].reshape(-1, 1),
                pred_probs[:, missing_class:],
            ),
            axis=1,
        )

    pred_probs = pred_probs / pred_probs.sum(axis=1)[:, np.newaxis]

    return pred_probs


def setup_next_iter_data(
    multiannotator_labels,
    images,
    images_unlabeled,
    pred_probs,
    pred_probs_unlabeled,
    true_labels,
    true_labels_unlabeled,
    extra_labels,
    extra_labels_unlabeled,
    quality_of_consensus,
    quality_of_consensus_unlabeled,
    num_annotators_to_add,
):

    num_labeled = len(images)

    quality_of_consensus_combine = np.concatenate(
        (quality_of_consensus, quality_of_consensus_unlabeled)
    )

    min_ind = np.argsort(quality_of_consensus_combine)[:num_annotators_to_add]
    min_ind_relabeled = min_ind[min_ind < num_labeled]
    min_ind_unlabeled = min_ind[min_ind >= num_labeled]

    images_new = images_unlabeled[min_ind_unlabeled - num_labeled]
    true_labels_new = true_labels_unlabeled[min_ind_unlabeled - num_labeled]
    pred_probs_new = pred_probs_unlabeled[min_ind_unlabeled - num_labeled, :]
    extra_labels_new = extra_labels_unlabeled[min_ind_unlabeled - num_labeled, :]

    images_unlabeled = np.delete(images_unlabeled, min_ind_unlabeled - num_labeled)
    true_labels_unlabeled = np.delete(
        true_labels_unlabeled, min_ind_unlabeled - num_labeled
    )
    extra_labels_unlabeled = np.delete(
        extra_labels_unlabeled, min_ind_unlabeled - num_labeled, axis=0
    )

    images = np.concatenate((images, images_new))
    true_labels = np.concatenate((true_labels, true_labels_new))
    pred_probs = np.concatenate((pred_probs, pred_probs_new))
    extra_labels = np.concatenate((extra_labels, extra_labels_new))
    idx_to_annotate = np.concatenate(
        (
            min_ind_relabeled,
            np.array(range(num_labeled, num_labeled + len(min_ind_unlabeled))),
        )
    ).astype(int)

    multiannotator_labels = pd.concat(
        (
            multiannotator_labels,
            pd.DataFrame(
                np.full(
                    (len(min_ind_unlabeled), multiannotator_labels.shape[1]), np.NaN
                )
            ),
        ),
        ignore_index=True,
    )

    num_new_examples = len(min_ind_unlabeled)

    return (
        multiannotator_labels,
        images,
        images_unlabeled,
        pred_probs,
        true_labels,
        true_labels_unlabeled,
        extra_labels,
        extra_labels_unlabeled,
        idx_to_annotate,
        num_new_examples,
    )


def setup_next_iter_data_ensemble(
    multiannotator_labels,
    images,
    images_unlabeled,
    pred_probs,
    pred_probs_unlabeled,
    true_labels,
    true_labels_unlabeled,
    extra_labels,
    extra_labels_unlabeled,
    quality_of_consensus,
    quality_of_consensus_unlabeled,
    num_annotators_to_add,
):

    num_labeled = len(images)

    quality_of_consensus_combine = np.concatenate(
        (quality_of_consensus, quality_of_consensus_unlabeled)
    )

    min_ind = np.argsort(quality_of_consensus_combine)[:num_annotators_to_add]
    min_ind_relabeled = min_ind[min_ind < num_labeled]
    min_ind_unlabeled = min_ind[min_ind >= num_labeled]

    images_new = images_unlabeled[min_ind_unlabeled - num_labeled]
    true_labels_new = true_labels_unlabeled[min_ind_unlabeled - num_labeled]
    pred_probs_new = pred_probs_unlabeled[:, min_ind_unlabeled - num_labeled, :]
    extra_labels_new = extra_labels_unlabeled[min_ind_unlabeled - num_labeled, :]

    images_unlabeled = np.delete(images_unlabeled, min_ind_unlabeled - num_labeled)
    true_labels_unlabeled = np.delete(
        true_labels_unlabeled, min_ind_unlabeled - num_labeled
    )
    extra_labels_unlabeled = np.delete(
        extra_labels_unlabeled, min_ind_unlabeled - num_labeled, axis=0
    )

    images = np.concatenate((images, images_new))
    true_labels = np.concatenate((true_labels, true_labels_new))
    pred_probs = np.hstack((pred_probs, pred_probs_new))
    extra_labels = np.concatenate((extra_labels, extra_labels_new))
    idx_to_annotate = np.concatenate(
        (
            min_ind_relabeled,
            np.array(range(num_labeled, num_labeled + len(min_ind_unlabeled))),
        )
    ).astype(int)

    multiannotator_labels = pd.concat(
        (
            multiannotator_labels,
            pd.DataFrame(
                np.full(
                    (len(min_ind_unlabeled), multiannotator_labels.shape[1]), np.NaN
                )
            ),
        ),
        ignore_index=True,
    )

    num_new_examples = len(min_ind_unlabeled)

    return (
        multiannotator_labels,
        images,
        images_unlabeled,
        pred_probs,
        true_labels,
        true_labels_unlabeled,
        extra_labels,
        extra_labels_unlabeled,
        idx_to_annotate,
        num_new_examples,
    )


def get_empirical_dist_entropy(multiannotator_labels):
    unique_ma_labels = np.unique(multiannotator_labels)
    unique_ma_labels = unique_ma_labels[~np.isnan(unique_ma_labels)]
    num_classes = len(unique_ma_labels)

    empirical_label_distribution = np.vstack(
        multiannotator_labels.apply(
            lambda s: np.bincount(s.dropna(), minlength=num_classes) / len(s.dropna()),
            axis=1,
        )
    )
    norm_entropy = get_normalized_entropy(
        np.array([np.mean(empirical_label_distribution, axis=0)])
    )
    return norm_entropy[0]


def add_new_annotator(multiannotator_labels, complete_labels, idxs):
    def get_random_label(annotator_labels):
        annotator_labels = annotator_labels[~np.isnan(annotator_labels)]
        return np.random.choice(annotator_labels)

    complete_labels_subset = complete_labels[idxs]
    new_annotator_labels = np.apply_along_axis(
        get_random_label, axis=1, arr=complete_labels_subset
    )

    # create new column
    new_annotator = np.full(len(multiannotator_labels), np.nan)
    new_annotator[idxs] = new_annotator_labels

    new_idx = np.max(list(multiannotator_labels.columns)) + 1
    multiannotator_labels[new_idx] = new_annotator

    return multiannotator_labels


def compute_soft_cross_entropy(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.ndarray,
) -> float:
    num_classes = get_num_classes(pred_probs=pred_probs)

    empirical_label_distribution = np.full(
        (len(labels_multiannotator), num_classes), np.NaN
    )
    for i in range(len(labels_multiannotator)):
        s = labels_multiannotator.iloc[i]
        empirical_label_distribution[i, :] = value_counts(
            s.dropna(), num_classes=num_classes
        ) / len(s.dropna())

    clipped_pred_probs = np.clip(pred_probs, a_min=1e-6, a_max=None)
    soft_cross_entropy = -np.sum(
        empirical_label_distribution * np.log(clipped_pred_probs), axis=1
    ) / np.log(num_classes)

    return soft_cross_entropy


def find_best_temp_scaler(
    labels_multiannotator: pd.DataFrame,
    pred_probs: np.ndarray,
) -> float:
    grid_search_coarse_range = np.array([0.1, 0.2, 0.5, 0.8, 1, 2, 3, 5, 8])
    soft_cross_entropy_coarse = np.full(len(grid_search_coarse_range), np.NaN)
    for i in range(len(grid_search_coarse_range)):
        curr_temp = grid_search_coarse_range[i]
        log_pred_probs = np.log(pred_probs) / curr_temp
        scaled_pred_probs = np.exp(log_pred_probs) / np.sum(
            np.exp(log_pred_probs)
        )  # softmax
        soft_cross_entropy_coarse[i] = np.mean(
            compute_soft_cross_entropy(labels_multiannotator, scaled_pred_probs)
        )

    min_entropy_ind = np.argmin(soft_cross_entropy_coarse)

    grid_search_fine_range = np.array([])
    if min_entropy_ind != 0:
        grid_search_fine_range = np.append(
            np.linspace(
                grid_search_coarse_range[min_entropy_ind - 1],
                grid_search_coarse_range[min_entropy_ind],
                4,
                endpoint=False,
            ),
            grid_search_fine_range,
        )
    if min_entropy_ind != len(grid_search_coarse_range) - 1:
        grid_search_fine_range = np.append(
            grid_search_fine_range,
            np.linspace(
                grid_search_coarse_range[min_entropy_ind],
                grid_search_coarse_range[min_entropy_ind + 1],
                5,
                endpoint=True,
            ),
        )
    soft_cross_entropy_fine = np.full(len(grid_search_fine_range), np.NaN)
    for i in range(len(grid_search_fine_range)):
        curr_temp = grid_search_fine_range[i]
        log_pred_probs = np.log(pred_probs) / curr_temp
        scaled_pred_probs = np.exp(log_pred_probs) / np.sum(
            np.exp(log_pred_probs)
        )  # softmax
        soft_cross_entropy_fine[i] = np.mean(
            compute_soft_cross_entropy(labels_multiannotator, scaled_pred_probs)
        )

    best_temp = grid_search_fine_range[np.argmin(soft_cross_entropy_fine)]

    return best_temp


def temp_scale_pred_probs(
    pred_probs: np.ndarray,
    temp: float,
) -> np.ndarray:
    log_pred_probs = np.log(pred_probs) / temp
    scaled_pred_probs = np.exp(log_pred_probs) / np.sum(
        np.exp(log_pred_probs)
    )  # softmax
    scaled_pred_probs = (
        scaled_pred_probs / np.sum(scaled_pred_probs, axis=1)[:, np.newaxis]
    )  # normalize

    return scaled_pred_probs
