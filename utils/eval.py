import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def get_annotator_mask(c10h_labels):
    mask = c10h_labels.copy()
    mask[~np.isnan(mask)] = 1
    mask[np.isnan(mask)] = 0
    return mask.astype(bool)

def get_labels_error_mask(c10h_labels, c10h_true_labels):
    tile_true_labels = np.repeat(c10h_true_labels[:,np.newaxis], c10h_labels.shape[1], axis=1)
    c10h_labels_error_mask = (c10h_labels == tile_true_labels) + 0.
    c10h_annotator_mask =  get_annotator_mask(c10h_labels)
    c10h_labels_error_mask = np.logical_and(c10h_labels_error_mask,c10h_annotator_mask)
    return c10h_labels_error_mask.astype(bool)

# Get accuracy of individual annotators on the points they labeled
def plt_annotator_accuracy(labels_error_mask, annotator_mask, plot=True, fig_title="fig.pdf"):
    annotator_accuracy = labels_error_mask.sum(axis=0) / annotator_mask.sum(axis=0)
    if plot:
        acc = pd.DataFrame(annotator_accuracy, columns=[''])
        bplot = acc.boxplot(figsize=(7,7), grid=False, fontsize=15)
        bplot.set_ylabel('Annotator Accuracy', fontsize=15)
        bplot.get_figure().gca().set_title("")
        bplot.get_figure().gca().set_xlabel("")
        plt.suptitle('')
        plt.savefig(fig_title, format="pdf")
        plt.show()

    df_describe = pd.DataFrame(annotator_accuracy, columns=['score'])
    return df_describe

# Plots the distribution of annotator agreement for correct/incorrect labels
def plt_labels_multiannotator(multiannotator_labels, consensus_labels, true_labels, plot=True, fig_title="fig"):
    consensus_labels_tile = np.repeat(consensus_labels[:,np.newaxis], multiannotator_labels.shape[1], axis=1)
    num_annotators_per_ex = np.count_nonzero(~np.isnan(multiannotator_labels), axis=1)
    
    annotator_agreement = (multiannotator_labels == consensus_labels_tile) # Number of annotators matches consensus
    annotator_agreement = annotator_agreement.sum(axis=1)
    
    bin_consensus = (true_labels == consensus_labels) + 0
    consensus_accuracy = pd.DataFrame(zip(annotator_agreement,bin_consensus), columns=['annotator agreement','binary consensus'])
    consensus_accuracy['binary consensus'] = consensus_accuracy.apply(lambda row: "Correct" if row['binary consensus']==1 else "Wrong", axis=1)
    
    if plot:
        bplot = consensus_accuracy.boxplot(by=['binary consensus'], figsize=(7,7), grid=False, fontsize=15)
        bplot.set_ylabel('Number of annotations for example', fontsize=15)
        bplot.set_xlabel('Consensus label accuracy', fontsize=15)
        bplot.get_figure().gca().set_title("")
        plt.suptitle('')
        plt.savefig(fig_title, format="pdf")
        plt.show()
    
    consensus_accuracy = pd.DataFrame(zip(annotator_agreement,bin_consensus), columns=['annotator agreement','binary consensus'])
    consensus_accuracy = consensus_accuracy.groupby('binary consensus')[['annotator agreement']].sum().reset_index()

    return consensus_accuracy

# Report how much the consensus labels and model predictions differ from true labels
def get_model_vs_consensus_accuracy(pred_probs, consensus_labels, true_labels):
    model_pred_labels = np.argmax(pred_probs, axis=1)
    rows = ['Annotator accuracy', 
            'Model accuracy', 
            'Shared labels between model and consensus']
    vals = [np.mean(true_labels == consensus_labels), np.mean(model_pred_labels == true_labels), np.mean(model_pred_labels == consensus_labels)]
    return pd.DataFrame(zip(rows,vals), columns=['Type', 'Percent'])