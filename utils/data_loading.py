from cleanlab.multiannotator import get_consensus_label
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, "../")

PATH = os.getcwd()
PATH = '/'.join(PATH.split('/')[:-1])

def get_annotator_labels(infolder):
    c10h_df = pd.read_csv(infolder)
    c10h_df = c10h_df[c10h_df.cifar10_test_test_idx != -99999] # dropping all attention check trials
    
    # initialize arrays
    c10h_num_datapoints = c10h_df['cifar10_test_test_idx'].max() + 1
    c10h_num_annotators = c10h_df['annotator_id'].max() + 1

    # get annotator labels as numpy array (N=labels, M=annotators)
    c10h_labels = np.full((c10h_num_datapoints, c10h_num_annotators), np.nan) # all annotator labels np.full([height, width, 9], np.nan)
    for annotator_id in range(c10h_num_annotators):
        adf = c10h_df[c10h_df.annotator_id == annotator_id] # 200 annotations per annotator
        annotations_idx = adf['cifar10_test_test_idx'].values
        annotations = adf['chosen_label'].values

        c10h_labels[annotations_idx, annotator_id] = annotations

    # get true labels as numpy array (N = true labels,)
    c10h_true_labels = np.zeros((c10h_num_datapoints, ))
    c10h_true_images = np.empty((c10h_num_datapoints, ) ,dtype=object)
    
    idx_to_label = \
    [(idx,label,image) for idx,label,image in zip(c10h_df['cifar10_test_test_idx'],c10h_df['true_label'],c10h_df['image_filename'])]
    idx_to_label = list(set(idx_to_label))

    idx = [idx_to_label[0] for idx_to_label in idx_to_label]
    true_label = [idx_to_label[1] for idx_to_label in idx_to_label]
    htrue_image = [idx_to_label[2] for idx_to_label in idx_to_label]

    c10h_true_labels[idx] = true_label
    c10h_true_images[idx] = htrue_image
    
    return c10h_labels, c10h_true_labels, c10h_true_images

# Returns sample labels/annotator_mask where x_drop, y_drop are idxs that are dropped
def _get_sample_labels(x_drop, y_drop, labels, annotator_mask):
    s_annotator_mask = annotator_mask.copy()
    s_annotator_mask[(x_drop,y_drop)] = 0
    s_labels = labels.copy()
    np.copyto(s_labels, np.nan, where=(s_annotator_mask==0)) 
    print('Total idxs dropped: ', annotator_mask.sum() - s_annotator_mask.sum())
    return s_labels, s_annotator_mask

# Returns a list of labeled indices to drop (random length per row)
def _get_random_drop_per_row(c10h_annotator_mask):
    x,y = np.where(c10h_annotator_mask == 1)
    idx_df = pd.DataFrame(zip(x,y),columns=['x','y'])
    for x_idx in range(idx_df['x'].max()+1):
        num_keep = np.random.randint(1, len(idx_df[idx_df['x'] == x_idx])+1)
        idx_df = idx_df.drop(idx_df[idx_df['x'] == x_idx].sample(num_keep).index)
    x_drop = idx_df['x'].values
    y_drop = idx_df['y'].values
    return x_drop, y_drop

# Returns a list of labeled indices to drop 
# (Randomly drop until <= max_annotations per example. Try to minimize number of distinct annotators)
def _get_random_drop_per_row_min_annotators(c10h_annotator_mask, max_annotations):
    x,y = np.where(c10h_annotator_mask == 1)
    xy = set([(x_idx,y_idx) for x_idx,y_idx in zip(x,y)])
    idx_df = pd.DataFrame(zip(x,y),columns=['x','y'])
    idx_keep = []
    selected_annotators = set()
    for x_idx in range(idx_df['x'].max()+1):
        Y = idx_df[idx_df['x'] == x_idx]['y']
        seen_annotators = set(Y).intersection(selected_annotators)
        if len(seen_annotators) < max_annotations: # We need to randomly select more annotators to greedy add
            num_to_find = max_annotations - len(seen_annotators)
            y_keep = set(np.random.choice(list(set(Y).difference(seen_annotators)), num_to_find, replace=False))
            selected_annotators = selected_annotators.union(y_keep)
            y_keep = seen_annotators.union(y_keep)
        else: # We have enough annotators and need to randomly select annotations out of the guys we have
            y_keep = np.random.choice(list(seen_annotators), max_annotations,replace=False)
        xy_keep = [(x_idx,y) for y in y_keep]
        idx_keep.extend(xy_keep)
    xy = xy.difference(set(idx_keep))
    x_drop = [xy_idx[0] for xy_idx in xy]
    y_drop = [xy_idx[1] for xy_idx in xy]
    return x_drop, y_drop

def _get_annotator_mask(c10h_labels):
    mask = c10h_labels.copy()
    mask[~np.isnan(mask)] = 1
    mask[np.isnan(mask)] = 0
    return mask

def drop_and_distribute(c10h_labels, max_annotations=5):
    c10h_annotator_mask = _get_annotator_mask(c10h_labels)
    x_drop, y_drop = _get_random_drop_per_row_min_annotators(c10h_annotator_mask, max_annotations)
    c10h_labels, c10h_annotator_mask = _get_sample_labels(x_drop, y_drop, c10h_labels, c10h_annotator_mask)
    print(f'Make sure {c10h_annotator_mask.sum(axis=1).max()} <= {n} and { c10h_annotator_mask.sum(axis=1).min()} > 0: ')

    x_drop, y_drop = _get_random_drop_per_row(c10h_annotator_mask)
    c10h_labels, c10h_annotator_mask = _get_sample_labels(x_drop, y_drop, c10h_labels, c10h_annotator_mask)
    print(f'Make sure {c10h_annotator_mask.sum(axis=1).max()} <= {n} and { c10h_annotator_mask.sum(axis=1).min()} > 0: ')

    # drop all empty annotators
    drop_axis = c10h_labels.copy()
    c10h_labels = c10h_labels[:, ~np.isnan(drop_axis).all(axis=0)]
    return c10h_labels

def get_and_save_consensus_labels(c10h_labels, c10h_true_labels, consensus_outfolder, pred_probs=None):
    classes = {0:"airplane", 
           1:"automobile", 
           2:"bird", 
           3:"cat", 
           4:"deer",
           5:"dog", 
           6:"frog", 
           7:"horse", 
           8:"ship", 
           9:"truck"}
    image_locs = [PATH + '/data/cifar10/test/' + classes[c10h_true_labels[i]] + 
                   '/test_batch_index_' + str(i).zfill(4) +'.png' for i in range(len(c10h_true_labels))]
    consensus_labels = get_consensus_label(pd.DataFrame(c10h_labels), pred_probs=pred_probs)
    consensus_df = pd.DataFrame(zip(image_locs, consensus_labels), columns=['image', 'label'])
    consensus_df.to_csv(consensus_outfolder, index=False)
    return consensus_df

def get_ground_truth_data_matched(model_infolder, annotator_infolder):
    # read numpy files from model_train_pred
    images = np.load(f"{infolder}/images.npy", allow_pickle=True)
    idxs = [int(image.split('/')[-1][-8:-4]) for image in images]
    
    # set all cifar10h annotator data to the correct indexing
    c10h_labels = np.load(f"{annotator_infolder}/c10h_labels.npy")
    c10h_true_labels = np.load(f"{annotator_infolder}/c10h_true_labels.npy")

    c10h_labels = c10h_labels[idxs]
    c10h_true_labels = c10h_true_labels[idxs]

    pred_probs = np.load(f"processed_data/{model}/pred_probs.npy")
    
    return c10h_labels, c10h_true_labels, pred_probs