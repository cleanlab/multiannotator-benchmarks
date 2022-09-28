# Multiannotator Benchmark

Benchmarking methods for the analysis of classification data labeled by multiple annotators.

This repository benchmarks algorithms that estimate:
1. A consensus label for each example that aggregates the individual annotations
2. A quality score for each consensus label which measures the confidence that a label is correct
3. An overall quality score for each annotator which measures the confidence in the overall correctness of labels obtained from this annotator

## Install Dependencies

To run the model training and benchmark you have to install the following dependencies:
```
pip install ./cleanlab
pip install ./crowd-kit
pip install -r requirements.txt
```

## Run Benchmarks

To reproduce the benchmarks using the results from the already trained models, run the following notebooks:

1. [benchmark.ipynb](2_benchmark.ipynb) - runs the benchmarks and saves results to csv
2. benchmark_results_[...].ipynb - visualize benchmark results in plots

## Generate Test Data and Train Model

To generate test data for your own experiments from scratch, run the following notebooks:

1. [preprocess_data.ipynb](0_preprocess_data.ipynb) - preprocesses CIFAR-10H dataset
2. [create_labels_df.ipynb](0_create_labels_df.ipynb) - generates correct absolute label paths for images in preprocessed data
3. [xval_model_train.ipynb](1_xval_model_train.ipynb) /  [xval_model_train_perfect_model.ipynb](1_xval_model_train_perfect_model.ipynb) - trains a model and generates test data
