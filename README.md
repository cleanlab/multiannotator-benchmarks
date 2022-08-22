# Multiannotator Benchmark
Benchmarking Cleanlab extension for multiannotator data labeling.

Our goal is to come up with algorithms that will:
1. Provide better consensus labels
2. Compute consensus quality scores 
3. Compute annotator quality scores

## Dataset Variants
Our primary dataset for the benchmark is the [CIFAR-10H](https://github.com/jcpeterson/cifar-10h) dataset, which contains multiple annotator labels for the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) test dataset.

We made additional variations of the CIFAR-10H dataset for our benchmarks, details of which are explained below:

#### 1. `complete`
This is the full, unaltered CIFAR-10H dataset, which contains 2571 annotators who each annotator 200 examples. Each example in this dataset has approximately 50 labels.

#### 2. `uniform_1_5`
To increase sparsity in the dataset, we subsetted the CIFAR10-H dataset such that each example now has between 1 to 5 labels (number of labels per example are uniformly distributed). After randomly dropping annotators and labels, our dataset contains 421 annotators.

#### 3. `worst_annotators`
This variation of the dataset selects the annotators that have the lowest accuracies when comparing their labels to the true labels. The sparsity and number of annotations per examples in this dataset is similar to `uniform_1_5` however the annotators are more likely to make label errors. This dataset contains 511 annotators.

## Evaluation Metrics

To measure the three goals stated above, we used the following metrics:

1. To evaluate **how well our methods can provide consensus labels**, we measure the accuracy of the consensus labels against the ground truth CIFAR-10 labels

2. To evaluate **how well the consensus quality scores estimate the quality of the consensus labels**, we measure the AUROC and AUPRC of whether or not the consensus label is accurate with respect to the ground truth label

3. To evaluate **how well the annotator quality scores estimate the quality of each annotator**, we meausre the spearman correlation of the annotator quality score and the accuracy of each annotator's labels with respect to the ground truth label



## Methods Benchmarked

Here is a brief description of all the methods we used in this benchmark.

| Method Name | Description |
| --- | ----------- |
| `crowdlab` | Weighted ensemble model that treats each annotator and the model as a predictor, weights for the ensemble are determined using the annotator quality scores | 
| `agreement` | Percentage of labels that agree with the consensus label |
| `label_quality_score` | cleanlab's `label_quality_score` of the consensus labels |
| `active_label_cleaning` | Ranks labels by noise, [link to paper](https://www.nature.com/articles/s41467-022-28818-3.pdf?origin=ppub) for more information |
| `no_perannotator_weights` | Similar idea to `crowdlab`, however all annotators are aggregated then weighed against the model |
| `empirical_bayes` | Treats the model prediction as prior distribution for true label and annotator labels as observations for a posterior distribution |
| `dawid_skene` |  An algorithm for crowdsourced vote aggregation based off the principle of Expectation-Maximization, [link to paper](https://www.jstor.org/stable/2346806?origin=crossref) for more information |
| `dawid_skene_with_model` | Adding the model predictions as an additional annotator, then running the Dawid-Skene algorithm above |
| `glad` | Uses probabilistic approaches to infer the label, the expertise of each annotator and the difficulty of each example, [link to paper](https://proceedings.neurips.cc/paper/2009/file/f899139df5e1059396431415e770c6dd-Paper.pdf) for more information |
| `glad_with_model` |  Adding the model predictions as an additional annotator, then running the GLAD algorithm above |
