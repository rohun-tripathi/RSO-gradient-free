This code is the training script for RSO and is tested using python 3.7.3, mxnet 1.6.0 and gluoncv (for the data sets)
Training runs on a GPU and using a GPU-compatible mxnet installation.

### Training
In the project home directory - 
1a) MNIST -> python train.py
1b) Depending on the GPU memory, reduce the batch size to train -> python train.py --sample_batch 2000

2) CIFAR_10 -> python train.py --data C10 --layers 9 --epochs 500

