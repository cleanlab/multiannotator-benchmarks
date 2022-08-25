 #! /bin/sh

 # downloads all data required for evaluate_benchmarks.ipynb
 pip install cifar2png
 cifar2png cifar10 ./data/cifar10 --name-with-batch-index
 cd data && git clone https://github.com/jcpeterson/cifar-10h.git
 cd cifar10h && mv data/* . rm -rf data && unzip cifar10h-raw
 unzip cifar10-raw.zip && rm cifar10h-raw.zip

