import sys

sys.path.insert(0, "../")

import numpy as np
import pandas as pd
import pickle
import datetime
from pathlib import Path
import cleanlab
from .cross_validation_autogluon import cross_val_predict_autogluon_image_dataset


def train_models(models, data_filepath, model_results_folder, num_cv_folds=5, verbose=1, epochs=1, holdout_frac=0.2, time_limit=60, random_state=123, **kwargs):    
#     # set xvalidation parameters and shareable model train params
#     num_cv_folds = kwargs['num_cv_folds'] if 'num_cv_folds' in kwargs.keys() else 5
#     verbose = kwargs['verbose'] if 'verbose' in kwargs.keys() else 1
#     epochs = kwargs['epochs'] if 'epochs' in kwargs.keys() else 1 #100
#     holdout_frac = kwargs['holdout_frac'] if 'holdout_frac' in kwargs.keys() else 0.2
#     time_limit = kwargs['time_limit'] if 'time_limit' in kwargs.keys() else 60 #21600
#     random_state = kwargs['random_state'] if 'random_state' in kwargs.keys() else 123
    
    # load data
    df = pd.read_csv(data_filepath)
    
    # run xvalidation for each model
    for model in models:
        print("----")
        print(f"Running cross-validation for model: {model}")

        MODEL_PARAMS = {
            "model": model,
            "epochs": epochs,
            "holdout_frac": holdout_frac,
        }

        # results of cross-validation will be saved to pickle files for each model/fold
        _ = \
            cross_val_predict_autogluon_image_dataset(
                dataset=df,
                out_folder=f"{model_results_folder}_{model}/", # save results of cross-validation in pickle files for each fold
                n_splits=num_cv_folds,
                model_params=MODEL_PARAMS,
                time_limit=time_limit,
                random_state=random_state,
            )

# load pickle file util
def _load_pickle(pickle_file_name, verbose=1):
    """Load pickle file"""
    if verbose:
        print(f"Loading {pickle_file_name}")
    with open(pickle_file_name, 'rb') as handle:
        out = pickle.load(handle)
    return out


def sum_xval_folds(models, model_results_folder, num_cv_folds=5, verbose=1, **kwargs):
    # get original label name to idx mapping
    label_name_to_idx_map = {'airplane': 0,
                         'automobile': 1,
                         'bird': 2,
                         'cat': 3,
                         'deer': 4,
                         'dog': 5,
                         'frog': 6,
                         'horse': 7,
                         'ship': 8,
                         'truck': 9}
    results_list = []

    for model in models:

        pred_probs = []
        labels = []
        images = []

        for split_num in range(num_cv_folds):

            out_subfolder = f"{model_results_folder}_{model}/split_{split_num}/"

            # pickle file name to read
            get_pickle_file_name = (
                lambda object_name: f"{out_subfolder}_{object_name}_split_{split_num}"
            )

            # NOTE: the "test_" prefix in the pickle name correspond to the "test" split during cross-validation.
            pred_probs_split = _load_pickle(get_pickle_file_name("test_pred_probs"), verbose=verbose)
            labels_split = _load_pickle(get_pickle_file_name("test_labels"), verbose=verbose)
            images_split = _load_pickle(get_pickle_file_name("test_image_files"), verbose=verbose)
            indices_split = _load_pickle(get_pickle_file_name("test_indices"), verbose=verbose)
            print(indices_split)
            
            # append to list so we can combine data from all the splits
            pred_probs.append(pred_probs_split)
            labels.append(labels_split)
            images.append(images_split)    

        # convert list to array
        pred_probs = np.vstack(pred_probs)
        labels = np.hstack(labels) # remember that this is the consensus labels (s)
        images = np.hstack(images)

        # get the original label from file path (aka "true labels" y)
        get_orig_label_idx_from_file_path = np.vectorize(lambda f: label_name_to_idx_map[Path(f).parts[-2]])
        true_labels = get_orig_label_idx_from_file_path(images)

        # save to Numpy files
        numpy_out_folder = f"{model_results_folder}_{model}/"

        print(f"Saving to numpy files in this folder: {numpy_out_folder}")

        np.save(numpy_out_folder + "pred_probs", pred_probs)
        np.save(numpy_out_folder + "labels", labels)
        np.save(numpy_out_folder + "images", images)
        np.save(numpy_out_folder + "true_labels", true_labels)

        return pred_probs, labels , true_labels, images