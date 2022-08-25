import numpy as np
from typing import List
import warnings
from sklearn.metrics import log_loss
from cleanlab.rank import get_label_quality_scores


def get_label_quality_ensemble_scores_experimental(
    labels: np.array,
    pred_probs_list: List[np.array],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    weight_ensemble_members_by: str = "accuracy",
    custom_weights: np.array = None,
    verbose: bool = True,
) -> np.array:
    """Returns label quality scores based on predictions from an ensemble of models.
    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.
    Ensemble scoring requires a list of pred_probs from each model in the ensemble.
    For each pred_probs in list, compute label quality score.
    Take the average of the scores with the chosen weighting scheme determined by `weight_ensemble_members_by`.
    Score is between 0 and 1:
    - 1 --- clean label (given label is likely correct).
    - 0 --- dirty label (given label is likely incorrect).
    Parameters
    ----------
    labels : np.array
      Labels in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
    pred_probs_list : List[np.array]
      Each element in this list should be an array of pred_probs in the same format
      expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
      Each element of `pred_probs_list` corresponds to the predictions from one model for all examples.
    method : {"self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default="self_confidence"
      Label quality scoring method. See :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>`
      for scenarios on when to use each method.
    adjust_pred_probs : bool, optional
      `adjust_pred_probs` in the same format expected by the :py:func:`get_label_quality_scores <cleanlab.rank.get_label_quality_scores>` function.
    weight_ensemble_members_by : {"uniform", "accuracy"}, default="accuracy"
      Weighting scheme used to aggregate scores from each model:
      - "uniform": take the simple average of scores
      - "accuracy": take weighted average of scores, weighted by model accuracy
    verbose : bool, default=True
      Set to ``False`` to suppress all print statements.
    Returns
    -------
    label_quality_scores : np.array
    See Also
    --------
    get_label_quality_scores
    """

    # Check pred_probs_list for errors
    assert isinstance(
        pred_probs_list, list
    ), f"pred_probs_list needs to be a list. Provided pred_probs_list is a {type(pred_probs_list)}"

    assert len(pred_probs_list) > 0, "pred_probs_list is empty."

    if len(pred_probs_list) == 1:
        warnings.warn(
            """
            pred_probs_list only has one element.
            Consider using get_label_quality_scores() if you only have a single array of pred_probs.
            """
        )

    # Generate scores for each model's pred_probs
    scores_list = []
    val_list = []
    for pred_probs in pred_probs_list:

        # Calculate scores and accuracy
        scores = get_label_quality_scores(
            labels=labels,
            pred_probs=pred_probs,
            method=method,
            adjust_pred_probs=adjust_pred_probs,
        )
        scores_list.append(scores)

        # Only compute if weighting by accuracy or inv-log-loss
        if weight_ensemble_members_by == "accuracy":
            accuracy = (pred_probs.argmax(axis=1) == labels).mean()
            val_list.append(accuracy)

        elif weight_ensemble_members_by == "inv_log_loss":
            inv_log_loss_ = 1 / log_loss(labels, pred_probs)
            val_list.append(inv_log_loss_)

    if verbose:
        print(f"Weighting scheme for ensemble: {weight_ensemble_members_by}")

    # Transform list of scores into an array of shape (N, M) where M is the number of models in the ensemble
    scores_ensemble = np.vstack(scores_list).T

    # Aggregate scores with chosen weighting scheme
    if weight_ensemble_members_by == "uniform":
        label_quality_scores = scores_ensemble.mean(
            axis=1
        )  # Uniform weights (simple average)

    elif weight_ensemble_members_by in ["accuracy", "inv_log_loss"]:
        weights = np.array(val_list) / sum(
            val_list
        )  # Weight by relative value (accuracy or inverse log loss)
        if verbose:
            print(
                f"Ensemble members will be weighted by: their relative {weight_ensemble_members_by}"
            )
            for i, val in enumerate(val_list):
                print(f"  Model {i} {weight_ensemble_members_by} : {val}")
                print(f"  Model {i} weights  : {weights[i]}")

        # Aggregate scores with weighted average
        label_quality_scores = (scores_ensemble * weights).sum(axis=1)

    elif weight_ensemble_members_by == "custom":

        print("Ensemble scoring using custom weights!")

        # Check that there is a weight per model
        assert len(custom_weights) == len(pred_probs_list)

        # Aggregate scores with weighted average
        label_quality_scores = (scores_ensemble * custom_weights).sum(axis=1)

    else:
        raise ValueError(
            f"""
            {weight_ensemble_members_by} is not a valid weighting method for weight_ensemble_members_by!
            Please choose a valid weight_ensemble_members_by: uniform, accuracy, inv_log_loss
            """
        )

    return label_quality_scores
