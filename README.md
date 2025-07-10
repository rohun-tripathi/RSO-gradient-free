RSO: A Gradient Free Sampling Based Approach For Training Deep Neural Networks - https://arxiv.org/abs/2005.05955

## Abstract
We propose RSO (random search optimization), a gradient free Markov Chain Monte Carlo search based approach for training deep neural networks. To this end, RSO adds a perturbation to a weight in a deep neural network and tests if it reduces the loss on a mini-batch. If this reduces the loss, the weight is updated, otherwise the existing weight is retained. Surprisingly, we find that repeating this process a few times for each weight is sufficient to train a deep neural network. The number of weight updates for RSO is an order of magnitude lesser when compared to backpropagation with SGD. RSO can make aggressive weight updates in each step as there is no concept of learning rate. The weight update step for individual layers is also not coupled with the magnitude of the loss. RSO is evaluated on classification tasks on MNIST and CIFAR-10 datasets with deep neural networks of 6 to 10 layers where it achieves an accuracy of 99.1% and 81.8% respectively. We also find that after updating the weights just 5 times, the algorithm obtains a classification accuracy of 98% on MNIST. 

This repository has the training Code for RSO.
Tested using python 3.7.3, mxnet 1.6.0 and gluoncv (for the data sets)
Training runs on a GPU and using a GPU-compatible mxnet installation.

## Training Params
In the project home directory - 
1a) MNIST -> python train.py
1b) Depending on the GPU memory, reduce the batch size to train -> python train.py --sample_batch 2000

2) CIFAR_10 -> python train.py --data C10 --layers 9 --epochs 500

## Cite
```
@misc{tripathi2020rsogradientfreesampling,
      title={RSO: A Gradient Free Sampling Based Approach For Training Deep Neural Networks}, 
      author={Rohun Tripathi and Bharat Singh},
      year={2020},
      eprint={2005.05955},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2005.05955}, 
}
```
