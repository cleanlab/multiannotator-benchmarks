import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_annotator_mask(c10h_labels):
    mask = c10h_labels.copy()
    mask[~np.isnan(mask)] = 1
    mask[np.isnan(mask)] = 0
    return mask.astype(bool)


def get_labels_error_mask(c10h_labels, c10h_true_labels):
    tile_true_labels = np.repeat(
        c10h_true_labels[:, np.newaxis], c10h_labels.shape[1], axis=1
    )
    c10h_labels_error_mask = (c10h_labels == tile_true_labels) + 0.0
    c10h_annotator_mask = get_annotator_mask(c10h_labels)
    c10h_labels_error_mask = np.logical_and(c10h_labels_error_mask, c10h_annotator_mask)
    return c10h_labels_error_mask.astype(bool)


# Get accuracy of individual annotators on the points they labeled
def plt_annotator_accuracy(
    labels_error_mask, annotator_mask, plot=True, fig_title="fig.pdf"
):
    annotator_accuracy = labels_error_mask.sum(axis=0) / annotator_mask.sum(axis=0)
    if plot:
        acc = pd.DataFrame(annotator_accuracy, columns=[""])
        bplot = acc.boxplot(figsize=(7, 7), grid=False, fontsize=15)
        bplot.set_ylabel("Annotator Accuracy", fontsize=15)
        bplot.get_figure().gca().set_title("")
        bplot.get_figure().gca().set_xlabel("")
        plt.suptitle("")
        plt.savefig(fig_title, format="pdf")
        plt.show()

    df_describe = pd.DataFrame(annotator_accuracy, columns=["score"])
    return df_describe
