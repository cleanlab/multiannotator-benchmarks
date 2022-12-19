import os
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier

from utils.model_training import train_cross_val_model
from utils.active_learning_utils import (
    handle_missing_classes,
    setup_next_iter_data,
    setup_next_iter_data_ensemble,
    get_empirical_dist_entropy,
    add_new_annotator,
    find_best_temp_scaler,
    temp_scale_pred_probs,
)
from cleanlab.multiannotator import (
    get_majority_vote_label,
    get_label_quality_multiannotator,
    get_label_quality_multiannotator_ensemble,
    get_majority_vote_label_ensemble,
)
from cleanlab.internal.label_quality_utils import get_normalized_entropy
from cleanlab.rank import get_label_quality_scores

import warnings

warnings.filterwarnings("ignore")

num_rounds = 5
num_iter = 20
num_annotators_to_add = 100
model_type = "mlp"


def get_data():
    multiannotator_labels = pd.DataFrame(
        np.load("data/multiannotator_labels_labeled.npy")
    )

    true_labels_labeled = np.load("data/true_labels_labeled.npy")
    true_labels_unlabeled = np.load("data/true_labels_unlabeled.npy")
    true_labels_test = np.load("data/true_labels_test.npy")

    extra_labels_labeled = np.load("data/extra_labels_labeled.npy")
    extra_labels_unlabeled = np.load("data/extra_labels_unlabeled.npy")

    X_labeled = np.load("data/X_labeled.npy")
    X_unlabeled = np.load("data/X_unlabeled.npy")
    X_test = np.load("data/X_test.npy")

    return (
        multiannotator_labels,
        true_labels_labeled,
        true_labels_unlabeled,
        true_labels_test,
        extra_labels_labeled,
        extra_labels_unlabeled,
        X_labeled,
        X_unlabeled,
        X_test,
    )


# entropy
def entropy():
    method_name = "entropy"
    for k in range(num_rounds):
        print(f"----- Running round {k} -----")

        (
            multiannotator_labels,
            true_labels_labeled,
            true_labels_unlabeled,
            true_labels_test,
            extra_labels_labeled,
            extra_labels_unlabeled,
            X_labeled,
            X_unlabeled,
            X_test,
        ) = get_data()

        accuracy_arr = np.full(num_iter, np.nan)
        model_accuracy_arr = np.full(num_iter, np.nan)
        distribution_entropy_arr = np.full(num_iter, np.nan)
        num_added_arr = np.full(num_iter, np.nan)
        per_example_count = []

        for i in range(num_iter):
            # print(f"----- Running iter {i} -----")
            if i == 0:
                consensus_labels = get_majority_vote_label(multiannotator_labels)
            else:
                consensus_labels = get_majority_vote_label(
                    multiannotator_labels, pred_probs
                )

            # get accuracies / stats for this iteration
            accuracy = np.mean(consensus_labels == true_labels_labeled)
            accuracy_arr[i] = accuracy

            empirical_dist_entropy = get_empirical_dist_entropy(multiannotator_labels)
            distribution_entropy_arr[i] = empirical_dist_entropy

            val_counts = multiannotator_labels.count(axis=1).to_numpy()
            full_val_counts = np.append(
                val_counts,
                np.full(
                    len(X_labeled) + len(X_unlabeled) - len(multiannotator_labels), 0
                ),
            )
            per_example_count.append(np.array(full_val_counts))

            # Train cross validation model
            (
                model_accuracy,
                pred_probs,
                pred_probs_unlabeled,
                pred_probs_test,
            ) = train_cross_val_model(
                MLPClassifier(max_iter=500),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            model_accuracy_arr[i] = model_accuracy

            quality_of_consensus = -get_normalized_entropy(pred_probs)

            if len(X_unlabeled) > 0:
                quality_of_consensus_unlabeled = -get_normalized_entropy(
                    pred_probs_unlabeled
                )

                (
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    idx_to_annotate,
                    num_new_examples,
                ) = setup_next_iter_data(
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    pred_probs_unlabeled,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    quality_of_consensus,
                    quality_of_consensus_unlabeled,
                    num_annotators_to_add,
                )

            else:
                idx_to_annotate = np.argsort(quality_of_consensus)[
                    :num_annotators_to_add
                ]
                num_new_examples = 0

            num_added_arr[i] = num_new_examples

            print(
                f"acc = {accuracy}, model_acc = {model_accuracy}, num_new={num_new_examples}"
            )

            multiannotator_labels = add_new_annotator(
                multiannotator_labels, extra_labels_labeled, idx_to_annotate
            )

        np.save(f"results/{model_type}/{method_name}_accuracy_{k}.npy", accuracy_arr)
        np.save(
            f"results/{model_type}/{method_name}_model_accuracy_{k}.npy",
            model_accuracy_arr,
        )
        np.save(
            f"results/{model_type}/{method_name}_distribution_entropy_{k}.npy",
            distribution_entropy_arr,
        )
        np.save(f"results/{model_type}/{method_name}_num_added_{k}.npy", num_added_arr)
        np.save(
            f"results/{model_type}/{method_name}_per_example_count_{k}.npy",
            np.array(per_example_count),
        )


# random
def random():
    method_name = "random"
    for k in range(num_rounds):
        print(f"----- Running round {k} -----")

        (
            multiannotator_labels,
            true_labels_labeled,
            true_labels_unlabeled,
            true_labels_test,
            extra_labels_labeled,
            extra_labels_unlabeled,
            X_labeled,
            X_unlabeled,
            X_test,
        ) = get_data()

        accuracy_arr = np.full(num_iter, np.nan)
        model_accuracy_arr = np.full(num_iter, np.nan)
        distribution_entropy_arr = np.full(num_iter, np.nan)
        num_added_arr = np.full(num_iter, np.nan)
        per_example_count = []

        for i in range(num_iter):
            # print(f"----- Running iter {i} -----")
            if i == 0:
                consensus_labels = get_majority_vote_label(multiannotator_labels)
            else:
                consensus_labels = get_majority_vote_label(
                    multiannotator_labels, pred_probs
                )

            # get accuracies / stats for this iteration
            accuracy = np.mean(consensus_labels == true_labels_labeled)
            accuracy_arr[i] = accuracy

            empirical_dist_entropy = get_empirical_dist_entropy(multiannotator_labels)
            distribution_entropy_arr[i] = empirical_dist_entropy

            val_counts = multiannotator_labels.count(axis=1).to_numpy()
            full_val_counts = np.append(
                val_counts,
                np.full(
                    len(X_labeled) + len(X_unlabeled) - len(multiannotator_labels), 0
                ),
            )
            per_example_count.append(np.array(full_val_counts))

            # Train cross validation model
            (
                model_accuracy,
                pred_probs,
                pred_probs_unlabeled,
                pred_probs_test,
            ) = train_cross_val_model(
                MLPClassifier(max_iter=500),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            model_accuracy_arr[i] = model_accuracy

            quality_of_consensus = np.random.rand(len(true_labels_labeled))

            if len(X_unlabeled) > 0:
                quality_of_consensus_unlabeled = np.random.rand(
                    len(true_labels_unlabeled)
                )

                (
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    idx_to_annotate,
                    num_new_examples,
                ) = setup_next_iter_data(
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    pred_probs_unlabeled,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    quality_of_consensus,
                    quality_of_consensus_unlabeled,
                    num_annotators_to_add,
                )

            else:
                idx_to_annotate = np.argsort(quality_of_consensus)[
                    :num_annotators_to_add
                ]
                num_new_examples = 0

            num_added_arr[i] = num_new_examples

            print(
                f"acc = {accuracy}, model_acc = {model_accuracy}, num_new={num_new_examples}"
            )

            multiannotator_labels = add_new_annotator(
                multiannotator_labels, extra_labels_labeled, idx_to_annotate
            )

        np.save(f"results/{model_type}/{method_name}_accuracy_{k}.npy", accuracy_arr)
        np.save(
            f"results/{model_type}/{method_name}_model_accuracy_{k}.npy",
            model_accuracy_arr,
        )
        np.save(
            f"results/{model_type}/{method_name}_distribution_entropy_{k}.npy",
            distribution_entropy_arr,
        )
        np.save(f"results/{model_type}/{method_name}_num_added_{k}.npy", num_added_arr)
        np.save(
            f"results/{model_type}/{method_name}_per_example_count_{k}.npy",
            np.array(per_example_count),
        )


# good random
def good_random():
    method_name = "good_random"
    for k in range(num_rounds):
        print(f"----- Running round {k} -----")

        (
            multiannotator_labels,
            true_labels_labeled,
            true_labels_unlabeled,
            true_labels_test,
            extra_labels_labeled,
            extra_labels_unlabeled,
            X_labeled,
            X_unlabeled,
            X_test,
        ) = get_data()

        accuracy_arr = np.full(num_iter, np.nan)
        model_accuracy_arr = np.full(num_iter, np.nan)
        distribution_entropy_arr = np.full(num_iter, np.nan)
        num_added_arr = np.full(num_iter, np.nan)
        per_example_count = []

        for i in range(num_iter):
            # print(f"----- Running iter {i} -----")
            if i == 0:
                consensus_labels = get_majority_vote_label(multiannotator_labels)
            else:
                consensus_labels = get_majority_vote_label(
                    multiannotator_labels, pred_probs
                )

            # get accuracies / stats for this iteration
            accuracy = np.mean(consensus_labels == true_labels_labeled)
            accuracy_arr[i] = accuracy

            empirical_dist_entropy = get_empirical_dist_entropy(multiannotator_labels)
            distribution_entropy_arr[i] = empirical_dist_entropy

            val_counts = multiannotator_labels.count(axis=1).to_numpy()
            full_val_counts = np.append(
                val_counts,
                np.full(
                    len(X_labeled) + len(X_unlabeled) - len(multiannotator_labels), 0
                ),
            )
            per_example_count.append(np.array(full_val_counts))

            # Train cross validation model
            (
                model_accuracy,
                pred_probs,
                pred_probs_unlabeled,
                pred_probs_test,
            ) = train_cross_val_model(
                MLPClassifier(max_iter=500),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            model_accuracy_arr[i] = model_accuracy

            quality_of_consensus = np.random.rand(len(true_labels_labeled)) + val_counts

            if len(X_unlabeled) > 0:
                quality_of_consensus_unlabeled = np.random.rand(
                    len(true_labels_unlabeled)
                )

                (
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    idx_to_annotate,
                    num_new_examples,
                ) = setup_next_iter_data(
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    pred_probs_unlabeled,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    quality_of_consensus,
                    quality_of_consensus_unlabeled,
                    num_annotators_to_add,
                )

            else:
                idx_to_annotate = np.argsort(quality_of_consensus)[
                    :num_annotators_to_add
                ]
                num_new_examples = 0

            num_added_arr[i] = num_new_examples

            print(
                f"acc = {accuracy}, model_acc = {model_accuracy}, num_new={num_new_examples}"
            )

            multiannotator_labels = add_new_annotator(
                multiannotator_labels, extra_labels_labeled, idx_to_annotate
            )

        np.save(f"results/{model_type}/{method_name}_accuracy_{k}.npy", accuracy_arr)
        np.save(
            f"results/{model_type}/{method_name}_model_accuracy_{k}.npy",
            model_accuracy_arr,
        )
        np.save(
            f"results/{model_type}/{method_name}_distribution_entropy_{k}.npy",
            distribution_entropy_arr,
        )
        np.save(f"results/{model_type}/{method_name}_num_added_{k}.npy", num_added_arr)
        np.save(
            f"results/{model_type}/{method_name}_per_example_count_{k}.npy",
            np.array(per_example_count),
        )


# uncertainty
def uncertainty():
    method_name = "uncertainty"
    for k in range(num_rounds):
        print(f"----- Running round {k} -----")

        (
            multiannotator_labels,
            true_labels_labeled,
            true_labels_unlabeled,
            true_labels_test,
            extra_labels_labeled,
            extra_labels_unlabeled,
            X_labeled,
            X_unlabeled,
            X_test,
        ) = get_data()

        accuracy_arr = np.full(num_iter, np.nan)
        model_accuracy_arr = np.full(num_iter, np.nan)
        distribution_entropy_arr = np.full(num_iter, np.nan)
        num_added_arr = np.full(num_iter, np.nan)
        per_example_count = []

        for i in range(num_iter):
            # print(f"----- Running iter {i} -----")
            if i == 0:
                consensus_labels = get_majority_vote_label(multiannotator_labels)
            else:
                consensus_labels = get_majority_vote_label(
                    multiannotator_labels, pred_probs
                )

            # get accuracies / stats for this iteration
            accuracy = np.mean(consensus_labels == true_labels_labeled)
            accuracy_arr[i] = accuracy

            empirical_dist_entropy = get_empirical_dist_entropy(multiannotator_labels)
            distribution_entropy_arr[i] = empirical_dist_entropy

            val_counts = multiannotator_labels.count(axis=1).to_numpy()
            full_val_counts = np.append(
                val_counts,
                np.full(
                    len(X_labeled) + len(X_unlabeled) - len(multiannotator_labels), 0
                ),
            )
            per_example_count.append(np.array(full_val_counts))

            # Train cross validation model
            (
                model_accuracy,
                pred_probs,
                pred_probs_unlabeled,
                pred_probs_test,
            ) = train_cross_val_model(
                MLPClassifier(max_iter=500),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            model_accuracy_arr[i] = model_accuracy

            quality_of_consensus = -(1 - np.max(pred_probs, axis=1))

            if len(X_unlabeled) > 0:
                quality_of_consensus_unlabeled = -(
                    1 - np.max(pred_probs_unlabeled, axis=1)
                )

                (
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    idx_to_annotate,
                    num_new_examples,
                ) = setup_next_iter_data(
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    pred_probs_unlabeled,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    quality_of_consensus,
                    quality_of_consensus_unlabeled,
                    num_annotators_to_add,
                )

            else:
                idx_to_annotate = np.argsort(quality_of_consensus)[
                    :num_annotators_to_add
                ]
                num_new_examples = 0

            num_added_arr[i] = num_new_examples

            print(
                f"acc = {accuracy}, model_acc = {model_accuracy}, num_new={num_new_examples}"
            )

            multiannotator_labels = add_new_annotator(
                multiannotator_labels, extra_labels_labeled, idx_to_annotate
            )

        np.save(f"results/{model_type}/{method_name}_accuracy_{k}.npy", accuracy_arr)
        np.save(
            f"results/{model_type}/{method_name}_model_accuracy_{k}.npy",
            model_accuracy_arr,
        )
        np.save(
            f"results/{model_type}/{method_name}_distribution_entropy_{k}.npy",
            distribution_entropy_arr,
        )
        np.save(f"results/{model_type}/{method_name}_num_added_{k}.npy", num_added_arr)
        np.save(
            f"results/{model_type}/{method_name}_per_example_count_{k}.npy",
            np.array(per_example_count),
        )


# crowdlab
def crowdlab():
    method_name = "crowdlab"
    for k in range(num_rounds):
        print(f"----- Running round {k} -----")

        (
            multiannotator_labels,
            true_labels_labeled,
            true_labels_unlabeled,
            true_labels_test,
            extra_labels_labeled,
            extra_labels_unlabeled,
            X_labeled,
            X_unlabeled,
            X_test,
        ) = get_data()

        accuracy_arr = np.full(num_iter, np.nan)
        model_accuracy_arr = np.full(num_iter, np.nan)
        distribution_entropy_arr = np.full(num_iter, np.nan)
        num_added_arr = np.full(num_iter, np.nan)
        per_example_count = []

        for i in range(num_iter):
            # print(f"----- Running iter {i} -----")
            if i == 0:
                consensus_labels = get_majority_vote_label(multiannotator_labels)
            else:
                optimal_temp = find_best_temp_scaler(multiannotator_labels, pred_probs)
                pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)
                results = get_label_quality_multiannotator(
                    multiannotator_labels,
                    pred_probs,
                    return_annotator_stats=False,
                    return_detailed_quality=False,
                    return_weights=True,
                )
                consensus_labels = results["label_quality"]["consensus_label"]

            # get accuracies / stats for this iteration
            accuracy = np.mean(consensus_labels == true_labels_labeled)
            accuracy_arr[i] = accuracy

            empirical_dist_entropy = get_empirical_dist_entropy(multiannotator_labels)
            distribution_entropy_arr[i] = empirical_dist_entropy

            val_counts = multiannotator_labels.count(axis=1).to_numpy()
            full_val_counts = np.append(
                val_counts,
                np.full(
                    len(X_labeled) + len(X_unlabeled) - len(multiannotator_labels), 0
                ),
            )
            per_example_count.append(np.array(full_val_counts))

            # Train cross validation model
            (
                model_accuracy,
                pred_probs,
                pred_probs_unlabeled,
                pred_probs_test,
            ) = train_cross_val_model(
                MLPClassifier(max_iter=500),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            model_accuracy_arr[i] = model_accuracy

            if i == 0:
                optimal_temp = find_best_temp_scaler(multiannotator_labels, pred_probs)
                pred_probs = temp_scale_pred_probs(pred_probs, optimal_temp)
                results = get_label_quality_multiannotator(
                    multiannotator_labels,
                    pred_probs,
                    return_annotator_stats=False,
                    return_detailed_quality=False,
                    return_weights=True,
                )

            prior_quality_of_consensus = results["label_quality"][
                "consensus_quality_score"
            ]
            model_weight = results["model_weight"]
            annotator_weight = results["annotator_weight"]
            avg_annotator_weight = np.mean(annotator_weight)
            num_classes = pred_probs.shape[1]

            quality_of_consensus = np.full(len(prior_quality_of_consensus), np.nan)
            for n in range(len(quality_of_consensus)):
                annotator_labels = multiannotator_labels.iloc[i]
                quality_of_consensus[n] = np.average(
                    (prior_quality_of_consensus[n], 1 / num_classes),
                    weights=(
                        np.sum(annotator_weight[annotator_labels.notna()])
                        + model_weight,
                        avg_annotator_weight,
                    ),
                )

            if len(X_unlabeled) > 0:
                pred_probs_unlabeled = temp_scale_pred_probs(
                    pred_probs_unlabeled, optimal_temp
                )
                prior_quality_of_consensus_unlabeled = np.max(
                    pred_probs_unlabeled, axis=1
                )

                quality_of_consensus_unlabeled = np.average(
                    np.stack(
                        [
                            prior_quality_of_consensus_unlabeled,
                            np.full(
                                len(prior_quality_of_consensus_unlabeled),
                                1 / num_classes,
                            ),
                        ]
                    ),
                    weights=[model_weight, avg_annotator_weight],
                    axis=0,
                )

                (
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    idx_to_annotate,
                    num_new_examples,
                ) = setup_next_iter_data(
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    pred_probs_unlabeled,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    quality_of_consensus,
                    quality_of_consensus_unlabeled,
                    num_annotators_to_add,
                )

            else:
                idx_to_annotate = np.argsort(quality_of_consensus)[
                    :num_annotators_to_add
                ]
                num_new_examples = 0

            num_added_arr[i] = num_new_examples

            print(
                f"acc = {accuracy}, model_acc = {model_accuracy}, num_new={num_new_examples}"
            )

            multiannotator_labels = add_new_annotator(
                multiannotator_labels, extra_labels_labeled, idx_to_annotate
            )

        np.save(f"results/{model_type}/{method_name}_accuracy_{k}.npy", accuracy_arr)
        np.save(
            f"results/{model_type}/{method_name}_model_accuracy_{k}.npy",
            model_accuracy_arr,
        )
        np.save(
            f"results/{model_type}/{method_name}_distribution_entropy_{k}.npy",
            distribution_entropy_arr,
        )
        np.save(f"results/{model_type}/{method_name}_num_added_{k}.npy", num_added_arr)
        np.save(
            f"results/{model_type}/{method_name}_per_example_count_{k}.npy",
            np.array(per_example_count),
        )


# disagreement
def disagreement():
    method_name = "disagreement"
    for k in range(num_rounds):
        print(f"----- Running round {k} -----")

        (
            multiannotator_labels,
            true_labels_labeled,
            true_labels_unlabeled,
            true_labels_test,
            extra_labels_labeled,
            extra_labels_unlabeled,
            X_labeled,
            X_unlabeled,
            X_test,
        ) = get_data()

        accuracy_arr = np.full(num_iter, np.nan)
        model_accuracy_arr = np.full(num_iter, np.nan)
        distribution_entropy_arr = np.full(num_iter, np.nan)
        num_added_arr = np.full(num_iter, np.nan)
        per_example_count = []

        for i in range(num_iter):
            # print(f"----- Running iter {i} -----")
            if i == 0:
                consensus_labels = get_majority_vote_label(multiannotator_labels)
            else:
                consensus_labels = get_majority_vote_label(
                    multiannotator_labels, pred_probs
                )

            # get accuracies / stats for this iteration
            accuracy = np.mean(consensus_labels == true_labels_labeled)
            accuracy_arr[i] = accuracy

            empirical_dist_entropy = get_empirical_dist_entropy(multiannotator_labels)
            distribution_entropy_arr[i] = empirical_dist_entropy

            val_counts = multiannotator_labels.count(axis=1).to_numpy()
            full_val_counts = np.append(
                val_counts,
                np.full(
                    len(X_labeled) + len(X_unlabeled) - len(multiannotator_labels), 0
                ),
            )
            per_example_count.append(np.array(full_val_counts))

            # Train cross validation model
            (
                model_accuracy1,
                pred_probs1,
                pred_probs_unlabeled1,
                pred_probs_test1,
            ) = train_cross_val_model(
                MLPClassifier(max_iter=500),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            (
                model_accuracy2,
                pred_probs2,
                pred_probs_unlabeled2,
                pred_probs_test2,
            ) = train_cross_val_model(
                ExtraTreesClassifier(),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            (
                model_accuracy3,
                pred_probs3,
                pred_probs_unlabeled3,
                pred_probs_test3,
            ) = train_cross_val_model(
                KNeighborsClassifier(weights="distance"),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            # model_accuracy_arr[i] = model_accuracy1

            pred_probs_stack = np.array((pred_probs1, pred_probs2, pred_probs3))
            pred_probs_mean = np.mean(pred_probs_stack, axis=0)
            pred_probs = pred_probs_mean
            num_classes = pred_probs_mean.shape[1]
            quality_of_consensus_matrix = np.full(pred_probs_mean.shape, np.nan)

            for m in range(len(pred_probs_stack)):
                curr_pred_probs = pred_probs_stack[m]
                clipped_pred_probs = np.clip(curr_pred_probs, a_min=1e-6, a_max=None)
                soft_cross_entropy = -np.sum(
                    pred_probs_mean * np.log(clipped_pred_probs), axis=1
                ) / np.log(num_classes)
                quality_of_consensus_matrix[:, m] = -soft_cross_entropy

            quality_of_consensus = np.mean(quality_of_consensus_matrix, axis=1)

            pred_probs_test_stack = np.array(
                (pred_probs_test1, pred_probs_test2, pred_probs_test3)
            )
            pred_probs_test = np.mean(pred_probs_test_stack, axis=0)
            model_pred_labels = np.argmax(pred_probs_test, axis=1)
            model_accuracy = np.mean(model_pred_labels == true_labels_test)
            model_accuracy_arr[i] = model_accuracy

            if len(X_unlabeled) > 0:
                pred_probs_unlabeled_stack = np.array(
                    (
                        pred_probs_unlabeled1,
                        pred_probs_unlabeled2,
                        pred_probs_unlabeled3,
                    )
                )
                pred_probs_unlabeled_mean = np.mean(pred_probs_unlabeled_stack, axis=0)
                pred_probs_unlabeled = pred_probs_unlabeled_mean
                quality_of_consensus_unlabeled_matrix = np.full(
                    pred_probs_unlabeled_mean.shape, np.nan
                )

                for m in range(len(pred_probs_stack)):
                    curr_pred_probs_unlabeled = pred_probs_unlabeled_stack[m]
                    clipped_pred_probs_unlabeled = np.clip(
                        curr_pred_probs_unlabeled, a_min=1e-6, a_max=None
                    )
                    soft_cross_entropy_unlabeled = -np.sum(
                        pred_probs_unlabeled_mean
                        * np.log(clipped_pred_probs_unlabeled),
                        axis=1,
                    ) / np.log(num_classes)
                    quality_of_consensus_unlabeled_matrix[
                        :, m
                    ] = -soft_cross_entropy_unlabeled

                quality_of_consensus_unlabeled = np.mean(
                    quality_of_consensus_unlabeled_matrix, axis=1
                )

                (
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    idx_to_annotate,
                    num_new_examples,
                ) = setup_next_iter_data(
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs,
                    pred_probs_unlabeled,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    quality_of_consensus,
                    quality_of_consensus_unlabeled,
                    num_annotators_to_add,
                )

            else:
                idx_to_annotate = np.argsort(quality_of_consensus)[
                    :num_annotators_to_add
                ]
                num_new_examples = 0

            num_added_arr[i] = num_new_examples

            print(
                f"acc = {accuracy}, model_acc = {model_accuracy}, num_new={num_new_examples}"
            )

            multiannotator_labels = add_new_annotator(
                multiannotator_labels, extra_labels_labeled, idx_to_annotate
            )

        np.save(f"results/{model_type}/{method_name}_accuracy_{k}.npy", accuracy_arr)
        np.save(
            f"results/{model_type}/{method_name}_model_accuracy_{k}.npy",
            model_accuracy_arr,
        )
        np.save(
            f"results/{model_type}/{method_name}_distribution_entropy_{k}.npy",
            distribution_entropy_arr,
        )
        np.save(f"results/{model_type}/{method_name}_num_added_{k}.npy", num_added_arr)
        np.save(
            f"results/{model_type}/{method_name}_per_example_count_{k}.npy",
            np.array(per_example_count),
        )


# crowdlab ensemble
def crowdlab_ensemble():
    method_name = "crowdlab_ensemble"
    for k in range(num_rounds):
        print(f"----- Running round {k} -----")

        (
            multiannotator_labels,
            true_labels_labeled,
            true_labels_unlabeled,
            true_labels_test,
            extra_labels_labeled,
            extra_labels_unlabeled,
            X_labeled,
            X_unlabeled,
            X_test,
        ) = get_data()

        accuracy_arr = np.full(num_iter, np.nan)
        model_accuracy_arr = np.full(num_iter, np.nan)
        distribution_entropy_arr = np.full(num_iter, np.nan)
        num_added_arr = np.full(num_iter, np.nan)
        per_example_count = []

        for i in range(num_iter):
            # print(f"----- Running iter {i} -----")
            if i == 0:
                consensus_labels = get_majority_vote_label(multiannotator_labels)
            else:
                for p in range(len(pred_probs_stack)):
                    optimal_temp = find_best_temp_scaler(
                        multiannotator_labels, pred_probs_stack[p]
                    )
                    pred_probs_stack[p] = temp_scale_pred_probs(
                        pred_probs_stack[p], optimal_temp
                    )
                results = get_label_quality_multiannotator_ensemble(
                    multiannotator_labels,
                    pred_probs_stack,
                    return_annotator_stats=False,
                    return_detailed_quality=False,
                    return_weights=True,
                )
                consensus_labels = results["label_quality"]["consensus_label"]

            # get accuracies / stats for this iteration
            accuracy = np.mean(consensus_labels == true_labels_labeled)
            accuracy_arr[i] = accuracy

            empirical_dist_entropy = get_empirical_dist_entropy(multiannotator_labels)
            distribution_entropy_arr[i] = empirical_dist_entropy

            val_counts = multiannotator_labels.count(axis=1).to_numpy()
            full_val_counts = np.append(
                val_counts,
                np.full(
                    len(X_labeled) + len(X_unlabeled) - len(multiannotator_labels), 0
                ),
            )
            per_example_count.append(np.array(full_val_counts))

            # Train cross validation model
            (
                model_accuracy1,
                pred_probs1,
                pred_probs_unlabeled1,
                pred_probs_test1,
            ) = train_cross_val_model(
                MLPClassifier(max_iter=500),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            (
                model_accuracy2,
                pred_probs2,
                pred_probs_unlabeled2,
                pred_probs_test2,
            ) = train_cross_val_model(
                ExtraTreesClassifier(),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            (
                model_accuracy3,
                pred_probs3,
                pred_probs_unlabeled3,
                pred_probs_test3,
            ) = train_cross_val_model(
                KNeighborsClassifier(weights="distance"),
                X_labeled,
                consensus_labels,
                true_labels_test,
                X_test,
                cv_n_folds=5,
                X_unlabeled=X_unlabeled,
            )

            # model_accuracy_arr[i] = model_accuracy1

            pred_probs_stack = np.array((pred_probs1, pred_probs2, pred_probs3))
            pred_probs_mean = np.mean(pred_probs_stack, axis=0)

            if i == 0:
                for p in range(len(pred_probs_stack)):
                    optimal_temp = find_best_temp_scaler(
                        multiannotator_labels, pred_probs_stack[p]
                    )
                    pred_probs_stack[p] = temp_scale_pred_probs(
                        pred_probs_stack[p], optimal_temp
                    )
                results = get_label_quality_multiannotator_ensemble(
                    multiannotator_labels,
                    pred_probs_stack,
                    return_annotator_stats=False,
                    return_detailed_quality=False,
                    return_weights=True,
                )

            prior_quality_of_consensus = results["label_quality"][
                "consensus_quality_score"
            ]
            model_weight = results["model_weight"]
            annotator_weight = results["annotator_weight"]
            avg_annotator_weight = np.mean(annotator_weight)
            num_classes = pred_probs_mean.shape[1]

            quality_of_consensus = np.full(len(prior_quality_of_consensus), np.nan)
            for n in range(len(quality_of_consensus)):
                annotator_labels = multiannotator_labels.iloc[i]
                quality_of_consensus[n] = np.average(
                    (prior_quality_of_consensus[n], 1 / num_classes),
                    weights=(
                        np.sum(annotator_weight[annotator_labels.notna()])
                        + np.sum(model_weight),
                        avg_annotator_weight,
                    ),
                )

            pred_probs = np.average(pred_probs_stack, weights=model_weight, axis=0)

            pred_probs_test_stack = np.array(
                (pred_probs_test1, pred_probs_test2, pred_probs_test3)
            )
            pred_probs_test = np.average(
                pred_probs_test_stack, weights=model_weight, axis=0
            )
            model_pred_labels = np.argmax(pred_probs_test, axis=1)
            model_accuracy = np.mean(model_pred_labels == true_labels_test)
            model_accuracy_arr[i] = model_accuracy

            if len(X_unlabeled) > 0:
                pred_probs_unlabeled_stack = np.array(
                    (
                        pred_probs_unlabeled1,
                        pred_probs_unlabeled2,
                        pred_probs_unlabeled3,
                    )
                )
                pred_probs_unlabeled_mean = np.mean(pred_probs_unlabeled_stack, axis=0)
                pred_probs_unlabeled = pred_probs_unlabeled_mean
                quality_of_consensus_unlabeled_matrix = np.full(
                    pred_probs_unlabeled_mean.shape, np.nan
                )

                for p in range(len(pred_probs_unlabeled_stack)):
                    optimal_temp = find_best_temp_scaler(
                        multiannotator_labels, pred_probs_stack[p]
                    )
                    pred_probs_unlabeled_stack[p] = temp_scale_pred_probs(
                        pred_probs_unlabeled_stack[p], optimal_temp
                    )

                consensus_label_unlabeled = get_majority_vote_label_ensemble(
                    np.argmax(pred_probs_unlabeled_stack, axis=2).T,
                    pred_probs_unlabeled_stack,
                )
                modified_pred_probs_unlabeled = np.average(
                    np.concatenate(
                        (
                            pred_probs_unlabeled_stack,
                            np.full(
                                pred_probs_unlabeled_stack.shape[1:], 1 / num_classes
                            )[np.newaxis, :, :],
                        )
                    ),
                    weights=np.concatenate((model_weight, [avg_annotator_weight])),
                    axis=0,
                )

                quality_of_consensus_unlabeled = get_label_quality_scores(
                    consensus_label_unlabeled, modified_pred_probs_unlabeled
                )

                pred_probs_unlabeled = np.average(
                    pred_probs_unlabeled_stack, weights=model_weight, axis=0
                )

                (
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs_stack,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    idx_to_annotate,
                    num_new_examples,
                ) = setup_next_iter_data_ensemble(
                    multiannotator_labels,
                    X_labeled,
                    X_unlabeled,
                    pred_probs_stack,
                    pred_probs_unlabeled_stack,
                    true_labels_labeled,
                    true_labels_unlabeled,
                    extra_labels_labeled,
                    extra_labels_unlabeled,
                    quality_of_consensus,
                    quality_of_consensus_unlabeled,
                    num_annotators_to_add,
                )

            else:
                idx_to_annotate = np.argsort(quality_of_consensus)[
                    :num_annotators_to_add
                ]
                num_new_examples = 0

            num_added_arr[i] = num_new_examples

            print(
                f"acc = {accuracy}, model_acc = {model_accuracy}, num_new={num_new_examples}"
            )

            multiannotator_labels = add_new_annotator(
                multiannotator_labels, extra_labels_labeled, idx_to_annotate
            )

        np.save(f"results/{model_type}/{method_name}_accuracy_{k}.npy", accuracy_arr)
        np.save(
            f"results/{model_type}/{method_name}_model_accuracy_{k}.npy",
            model_accuracy_arr,
        )
        np.save(
            f"results/{model_type}/{method_name}_distribution_entropy_{k}.npy",
            distribution_entropy_arr,
        )
        np.save(f"results/{model_type}/{method_name}_num_added_{k}.npy", num_added_arr)
        np.save(
            f"results/{model_type}/{method_name}_per_example_count_{k}.npy",
            np.array(per_example_count),
        )