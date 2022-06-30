import numpy as np


def lift_at_k(y_true: np.array, y_score: np.array, k: int = 100) -> np.float:
    """Compute Lift at K evaluation metric"""

    # sort scores
    sort_indices = np.argsort(y_score)

    # compute lift for the top k values
    lift_at_k = y_true[sort_indices][-k:].mean() / y_true.mean()

    return lift_at_k
