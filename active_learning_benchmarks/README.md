# Benchmarking methods to select examples to relabel in active learning for data labeled by multiple annotators

This repository benchmarks algorithms that compute an active learning score that quantifies how desirable and valuable it is to collect additional labels for examples in a dataset.

This repository is only for intended for scientific purposes. To apply the CROWDLAB active learning algorithm to your own active learning loops with multiannotator data, you should instead use the implementation from the official [cleanlab](https://github.com/cleanlab/cleanlab) library.

## Install Dependencies

To run the model training and benchmarks, you need to install the following dependencies:

```
pip install -r requirements.txt
pip install cleanlab
```

## Benchmarks

Three sets of benchmarks are conducted with 3 different datasets: 

|   | Dataset | Description | 
| - | ------- | ----------- |
| 1 | [CIFAR-10H](cifar-10h) | Image classification with a total of 5000 examples, where 1000 examples have annotator labels at the beginning, we collect 500 new labels each round. |
| 2 | [Wall Robot](wall-robot) | Tabular classification with a total of 2000 examples, where 500 examples have annotator labels at the beginning, we collect 100 new labels each round. |
| 3 | [Wall Robot Complete](wall-robot-completely-labeled) | Tabular classification with a total of 2000 examples, where all 2000 examples have annotator labels at round 0, we collect 100 new labels each round. |

The datasets used in the benchmark are downloaded from:

- [CIFAR-10H](https://github.com/jcpeterson/cifar-10h)
- [Wall-Following Robot Navigation Data (Wall Robot)](https://www.openml.org/search?type=data&sort=runs&status=any&qualities.NumberOfClasses=gte_2&qualities.NumberOfInstances=between_1000_10000&id=1526)