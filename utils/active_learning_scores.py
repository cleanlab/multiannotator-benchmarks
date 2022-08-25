# active learning scores
import numpy as np


def least_confidence(
    pred_probs: np.array, labels: np.array = np.array(None), **kwargs
) -> np.array:
    """
    Calculate the difference between 1 and the most confident prediction.

    Least confidence is used in active learning for uncertainty sampling: https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b

    Scores range from 0 to 1.

    Parameters
    ----------
    pred_probs : ndarray of shape (n_samples, n_classes)
        Predicted probabilities for each class.
        These predicted probabilities need to be generated out-of-sample (e.g. via cross-validation).

    labels : None
        Not used, present here for API consistency by convention. Same API design as scikit-learn.

    Returns
    -------
    least_confidence : ndarray of shape (n_samples,)

    """

    num_classes = pred_probs.shape[1]
    max_confidence = pred_probs.max(axis=1)

    return (1.0 - max_confidence) * num_classes / (num_classes - 1)
