import numpy as np
import warnings
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from cleanlab.internal.util import (
    get_num_classes,
    append_extra_datapoint,
    train_val_split,
)


def train_cross_val_model(
    clf,
    X_train,
    consensus_label,
    true_labels_test,
    X_test,
    cv_n_folds=5,
    X_unlabeled=None,
):
    # run xvalidation
    # print(f"Running cross-validation for model: {clf}")

    num_classes = get_num_classes(labels=consensus_label)
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True)

    # Initialize pred_probs array
    pred_probs = np.zeros(shape=(len(consensus_label), num_classes))
    pred_probs_unlabeled = []
    pred_probs_test = []
    model_accuracy = []

    for k, (cv_train_idx, cv_holdout_idx) in enumerate(
        kf.split(X=X_train, y=consensus_label)
    ):
        clf_copy = sklearn.base.clone(clf)  # fresh untrained copy of the model

        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv, s_train_cv, s_holdout_cv = train_val_split(
            X_train, consensus_label, cv_train_idx, cv_holdout_idx
        )

        missing_class_inds = {}
        # Ensure no missing classes in training set.
        train_cv_classes = set(s_train_cv)
        all_classes = set(range(num_classes))
        if len(train_cv_classes) != len(all_classes):
            missing_classes = all_classes.difference(train_cv_classes)
            warnings.warn(
                "Duplicated some data across multiple folds to ensure training does not fail "
                f"because these classes do not have enough data for proper cross-validation: {missing_classes}."
            )
            for missing_class in missing_classes:
                # Duplicate one instance of missing_class from holdout data to the training data:
                holdout_inds = np.where(s_holdout_cv == missing_class)[0]
                dup_idx = holdout_inds[0]
                s_train_cv = np.append(s_train_cv, s_holdout_cv[dup_idx])
                # labels are always np.ndarray so don't have to consider .iloc above
                X_train_cv = append_extra_datapoint(
                    to_data=X_train_cv, from_data=X_holdout_cv, index=dup_idx
                )
                missing_class_inds[missing_class] = dup_idx

        # Fit classifier clf to training set, predict on holdout set, and update pred_probs.
        clf_copy.fit(X_train_cv, s_train_cv)
        pred_probs_cv = clf_copy.predict_proba(X_holdout_cv)  # P(labels = k|x) # [:,1]

        # Replace predictions for duplicated indices with dummy predictions:
        for missing_class in missing_class_inds:
            dummy_pred = np.zeros(pred_probs_cv[0].shape)
            dummy_pred[missing_class] = 1.0  # predict given label with full confidence
            dup_idx = missing_class_inds[missing_class]
            pred_probs_cv[dup_idx] = dummy_pred

        pred_probs[cv_holdout_idx] = pred_probs_cv

        if len(X_unlabeled) > 0:
            curr_pred_probs_unlabeled = clf_copy.predict_proba(X_unlabeled)
        else:
            curr_pred_probs_unlabeled = np.array([])

        curr_pred_probs_test = clf_copy.predict_proba(X_test)
        curr_model_pred_labels = clf_copy.predict(X_test)
        curr_model_accuracy = np.mean(curr_model_pred_labels == true_labels_test)

        pred_probs_unlabeled.append(curr_pred_probs_unlabeled)
        pred_probs_test.append(curr_pred_probs_test)
        model_accuracy.append(curr_model_accuracy)

    avg_model_accuacy = np.mean(np.array(model_accuracy))
    avg_pred_probs_unlabeled = np.mean(np.array(pred_probs_unlabeled), axis=0)
    avg_pred_probs_test = np.mean(np.array(pred_probs_test), axis=0)

    return avg_model_accuacy, pred_probs, avg_pred_probs_unlabeled, avg_pred_probs_test
