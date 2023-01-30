# Benchmarking methods for classification data labeled by multiple annotators

Code to reproduce results from the paper:

**[CROWDLAB: Supervised learning to infer consensus labels and quality scores for data with multiple annotators](https://arxiv.org/abs/2210.06812)**  
*NeurIPS 2022 Human in the Loop Learning Workshop*

This repository benchmarks algorithms that estimate:
1. A consensus label for each example that aggregates the individual annotations.
2. A confidence score for the correctness of each consensus label.
3. A rating for each annotator which estimates the overall correctness of their labels.

This repository is only for intended for scientific purposes. 
To apply the CROWDLAB algorithm to your own multi-annotator data, you should instead use [the implementation](https://docs.cleanlab.ai/stable/tutorials/multiannotator.html) from the official [cleanlab](https://github.com/cleanlab/cleanlab) library.


## Install Dependencies

To run the model training and benchmark, you need to install the following dependencies:
```
pip install ./cleanlab
pip install ./crowd-kit
pip install -r requirements.txt
```

Note that our `cleanlab/` and `crowd-kit/` folders here contain forks of the [cleanlab](https://github.com/cleanlab/cleanlab) and [crowd-kit](https://github.com/Toloka/crowd-kit) libraries. These forks differ from the main libraries as follows:

- The `cleanlab` fork contains various multi-annotator algorithms studied in the benchmark (to obtain consensus labels and compute consensus and annotator quality scores) that are not present in the main library.
- The `crowd-kit` fork addresses some numeric underflow issues in the original library (needed for properly ranking examples by their quality). Instead of operating directly on probabilities, our fork does calculations on log-probabilities with the log-sum-exp trick.

## Run Benchmarks

To benchmark various multi-annotator algorithms using given predictions from already trained classifier models, run the following notebooks:

1. [benchmark.ipynb](2_benchmark.ipynb) - runs the benchmarks and saves results to csv
2. benchmark_results_[...].ipynb - visualize benchmark results in plots

## Generate Data and Train Classfier Model

To generate the multi-annotator datasets and train the image classifier considered in our benchmarks, run the following notebooks:

1. [preprocess_data.ipynb](0_preprocess_data.ipynb) - preprocesses the dataset
2. [create_labels_df.ipynb](0_create_labels_df.ipynb) - generates correct absolute label paths for images in preprocessed data
3. [xval_model_train.ipynb](1_xval_model_train.ipynb) /  [xval_model_train_perfect_model.ipynb](1_xval_model_train_perfect_model.ipynb) - trains a model and obtains predicted class probabilities for each image
