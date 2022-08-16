#!/usr/bin/env python
# coding: utf-8

# # This notebook runs a train and eval loop on models with improving consensus labels over each iteration.

# In[1]:


import sys
import numpy as np
import os
sys.path.insert(0, "../")

from utils.model_training import train_models
from utils.model_training import sum_xval_folds
from utils.data_loading import get_annotator_labels
from utils.data_loading import drop_and_distribute
from utils.data_loading import get_ground_truth_data_matched
from utils.data_loading import get_and_save_improved_consensus_label
from utils.data_loading import get_and_save_consensus_labels
from cleanlab.multiannotator import get_label_quality_multiannotator # only in hui wen directory
from datetime import datetime


# In[2]:


now = datetime.now() # current date and time
experiment_folder = "experiment_" + str(int(now.timestamp()))
dirName = './data/experiments/' + experiment_folder

if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")

print(f'Experiment saved in {dirName}')


# ## Dropout data values

# In[3]:


# Get cifar10h dataset and dropout information from it
cifar10_infolder = './data/cifar-10h/cifar10h-raw.csv' #c10h raw data folder
max_annotations = 5

c10h_labels, c10h_true_labels, c10h_true_images = get_annotator_labels(cifar10_infolder)
c10h_labels = drop_and_distribute(c10h_labels, max_annotations)

# save c10h_results
cifar10_labels_folder = f"{dirName}/todelete_c10h_labels_range_{max_annotations}.npy"
cifar10_true_labels_folder = f"{dirName}/todelete_c10h_true_labels_range_{max_annotations}.npy"
np.save(cifar10_labels_folder, c10h_labels)
np.save(cifar10_true_labels_folder, c10h_true_labels)

# Generate and save original consensus labels
consensus_outfolder = f'{dirName}/todelete_cifar10_test_consensus_dataset_range_{max_annotations}_0.csv' #output folder for consensus labels
consensus_labels = get_and_save_consensus_labels(c10h_labels, c10h_true_labels, consensus_outfolder)

# Generate label quality of each annotator
label_quality_multiannotator = None


# ## Train models through loop

# In[4]:


# Load consensus labels and train model on them
models = [
    "resnet18",
#     "swin_base_patch4_window7_224"
]

train_args = {
    "num_cv_folds": 5, 
    "verbose": 1, 
    "epochs": 1, 
    "holdout_frac": 0.2, 
    "time_limit": 60, 
    "random_state": 123
}


# In[5]:


# Loop through and retrain model on better pred-probs
NUM_MODEL_RETRAINS = 10

for i in range(NUM_MODEL_RETRAINS):
    for model in models:
        # Get folders
        if i == 0:
            consensus_infolder = consensus_outfolder
        else:
            consensus_infolder = f'{dirName}/todelete_cifar10_test_consensus_dataset_range_{max_annotations}_{i-1}_{model}.csv'
        model_results_folder = f'{dirName}/todelete_cifar10_consensus_range_{max_annotations}_{i}' # + [model_type]
        consensus_outfolder = f'{dirName}/todelete_cifar10_test_consensus_dataset_range_{max_annotations}_{i}_{model}.csv'

        print(f'--INFO {i}_{model}--')
        print('Loading consensus from', consensus_infolder)
        print('Saving consensus to', consensus_outfolder)
        print('Saving model results to', model_results_folder)
        print('---------------------')
        
        # Train model
        train_models([model], consensus_infolder, model_results_folder, **train_args)
        pred_probs, labels , true_labels, images = sum_xval_folds([model], model_results_folder, **train_args)
        
        # Get label quality multiannotator
        label_quality_multiannotator = get_label_quality_multiannotator(c10h_labels,pred_probs,verbose=False) if label_quality_multiannotator is None else label_quality_multiannotator

        # Generate and save consensus labels
        _ = get_and_save_improved_consensus_label(label_quality_multiannotator, c10h_true_labels, consensus_outfolder)


# ## Compute accuracy of model based on Accuracy (labels vs true labels) by itter after folder

# In[6]:


acc_noisy_vs_true_labels = (consensus_labels['label'].values == c10h_true_labels).mean()
print(f"Accuracy ORIGINAL (consensus labels vs true labels): {acc_noisy_vs_true_labels}\n")

for model in models:
    for i in range(NUM_MODEL_RETRAINS):
        
        # Get folders
        if i == 0:
            consensus_infolder = consensus_outfolder
        else:
            consensus_infolder = f'{dirName}/todelete_cifar10_test_consensus_dataset_range_{max_annotations}_{i-1}_{model}.csv'
        model_results_folder = f'{dirName}/todelete_cifar10_consensus_range_{max_annotations}_{i}' # + [model_type]
    
        print(f'--{model} iter{i}--')
        
        out_subfolder = f"{model_results_folder}_{model}/"
        pred_probs = np.load(out_subfolder + "pred_probs.npy")
        labels = np.load(out_subfolder + "labels.npy") # remember that this is the noisy labels (s)
        images = np.load(out_subfolder + "images.npy", allow_pickle=True)
        true_labels = np.load(out_subfolder + "true_labels.npy")

        # check the accuracy
        acc_labels = (pred_probs.argmax(axis=1) == labels).mean() # noisy labels (s)
        acc_true_labels = (pred_probs.argmax(axis=1) == true_labels).mean() # true labels (y)    
        acc_noisy_vs_true_labels = (labels == true_labels).mean()

        print(f"Model: {model}")
        print(f"  Accuracy (argmax pred vs labels)                 : {acc_labels}")
        print(f"  Accuracy (argmax pred vs true labels)            : {acc_true_labels}")
        print(f"  Accuracy (consensus labels vs true labels)       : {acc_noisy_vs_true_labels}\n")


# In[ ]:





# In[ ]:




