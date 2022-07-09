``data`` should contain following folders:
- ``benchmark_results/`` (results of evaluate_benchmarks notebook)
- ``model_data/`` (data to train model, model prediction results)
- ``cifar10h/`` (cifar10h dataset, download help below)
- ``cifar10/`` (cifar10 png files dataset, download help below)

To run ``../evaluate_benchmarks.ipynb``, make sure both cifar10 pngs and cifar10h datasets are downloaded locally in this directory:
1. ``pip install cifar2png`` #install png installer for cifar10 images
2. ``cifar2png cifar10 ./cifar10 --name-with-batch-index`` #download the cifar10 png images
3. ``git clone https://github.com/jcpeterson/cifar-10h.git`` #download cifar-10h dataset
5. Unizip ``cifar10h-raw`` and extract files out of ``cifar10-raw`` folder into ``cifar-10h``
4. Run ``../create_labels_df.ipynb`` to update image paths
