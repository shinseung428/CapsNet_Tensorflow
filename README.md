# Testing Capsule Network on various datasets

This repository contains different tests performed on a capsule network model. 

[**Test 1 : Capsule Network on mnist dataset**](#test-1-mnist---mnist)  
[**Test 2 : Capsule Network on fashion_mnist dataset**](#test-2-fashion-mnist---fashion-mnist)  
[**Test 3 : Capsule Network on small_norb dataset**](#test-3-smallnorbrandom-crop---smallnorbcenter-crop)  
[**Test 3 : Capsule Network on cifar10 dataset**](#test-4-cifar10---cifar10)  
[**Test 4 : Robustness of Capsule Network on randomly rotated mnist datset**](#test-5-mnist---mnistrotated)  
[**Test 5 : Robustness of Capsule Network on affine transformation**](#test-6-mnist---affnist)  


## Available dataset

* [mnist](http://yann.lecun.com/exdb/mnist/)
* [fashion_mnist](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)
* [affnist](http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/)
* [small_norb](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
* [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)

### data folder setting
```
-data
  -mnist
    -t10k-images.idx3-ubyte
    -t10k-labels.idx1-ubyte
    -train-images.idx3-ubyte
    -train-labels.idx1-ubyte
  -fashion-mnist
    -fashion-mnist_test.csv
    -fashion-mnist_train.csv
    -t10k-images-idx3-ubyte
    -t10k-labels-idx1-ubyte
    -train-images-idx3-ubyte
    -train-labels-idx1-ubyte
  -affnist
    -test
      -1.mat
      -2.mat
      -...
    -train
      -1.mat
      -2.mat
      -...
  -small_norb
    -smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat
    -smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat
    -smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat
    -smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat
  -cifar10
    -batches.meta
    -data_batch_1
    -...
    -test_batch
```
## Network Model
* baseline_network (Convolutional Neural Network as described in CapsNet paper)
* capsule_dynamic (Capsule Network with Dynamic Routing)
* capsule_em *(Coming soon)*

## Requirements
* python 2.7
* Tensorflow 1.3
* scipy


## How to run Training & Testing
Example code to train the capsule_dynamic(CapsNet with dynamic routing) model on mnist dataset.
```
$ python main.py --model=capsule_dynamic --data=mnist
```

Example code to test the capsule_dynamic(CapsNet with dynamic routing) model on mnist dataset.
```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist
```

Other models can be trained/tested by changing the name of the --model flag, and other datasets can be used by changing the name of the --data flag.

## Test 1 (mnist -> mnist)
Code to run the test
```
$ python main.py --model=capsule_dynamic --data=mnist 
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist 
```

***Trained mnist images***
![Alt text](images/mnist_gt.jpg?raw=true "mnist")
***Reconstructed mnist images***
![Alt text](images/mnist_recon.jpg?raw=true "mnist reconstructed")

| Model            | Parameters | Test Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    13.2M   |  99.03 % |
| capsule_dynamic  |     8.2M   |  99.25 % |

Both baseline_network and capsule_dynamic network achieved above 99% accuracy on the mnist dataset. 

## Test 2 (fashion-mnist -> fashion-mnist)

Code to run the test
```
$ python main.py --model=capsule_dynamic --data=fashion-mnist 
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=fashion-mnist 
```

***Trained fashion-mnist images***
![Alt text](images/fashion_mnist_gt.jpg?raw=true "fashion-mnist")
***Reconstructed fashion-mnist images***
![Alt text](images/fashion_mnist_recon.jpg?raw=true "fashion-mnist reconstructed") 

| Model            | Parameters | Test Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    13.2M   |  89.94%  |
| capsule_dynamic  |     8.2M   |  89.02%  |


Both baseline_network and capsule_dynamic networks achieved about 89% accuracy on the fashion-mnist dataset. This result was obtained after training the network for 5 epochs. Higher accuracy is expected when the network is trained more than 5 epochs. 

## Test 3 (smallNORB(random crop) -> smallNORB(center crop))
**size of the input(in main.py) should be changed to 32x32 before running this test**  
Code to run the test
```
$ python main.py --model=capsule_dynamic --data=small_norb
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=small_norb
```

***Trained randomly cropped 32x32 small_norb images***
![Alt text](images/smallnorb_train.jpg?raw=true "smallnorb")
***Tested center cropped 32x32 small_norb images***
![Alt text](images/smallnorb_test.jpg?raw=true "smallnorb")

| Model            | Parameters | Test Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    19.3M   |  99.16%  |
| capsule_dynamic  |     8.3M   |  99.56%  |

## Test 4 (cifar10 -> cifar10)
**size of the input(in main.py) should be changed to 32x32 before running this test**  
Code to run the test
```
$ python main.py --model=capsule_dynamic --data=cifar10
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=cifar10
```

***Trained randomly cropped 32x32 cifar10 images***
![Alt text](images/cifar_train.jpg?raw=true "smallnorb")
***Tested center cropped 32x32 cifar10 images***
![Alt text](images/cifar_test.jpg?raw=true "smallnorb")

| Model            | Parameters | Test Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    19.3M   |  67.42%  |
| capsule_dynamic  |    11.7M   |  69.82%  |


## Test 5 (mnist -> mnist(rotated))
Code to run the test
```
$ python main.py --model=capsule_dynamic --data=mnist
```
```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist --rotate=True
```

***Trained mnist images***
![Alt text](images/mnist_gt.jpg?raw=true "mnist") 
***Tested randomly rotated mnist images***
![Alt text](images/mnist_rotated.jpg?raw=true "rotated mnist")

| Model            | Parameters | Test Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    13.2M   |  74.10 % |
| capsule_dynamic  |     8.2M   |  77.68 % |


The baseline_network(CNN) and capsule_dynamic(CapsNet with dynamic routing) models were first trained on the normal 28x28 mnist dataset. Both network achieved high accuracy on the test set (both close to 99%). 

Two models were then tested on randomly rotated(-30 to +30) mnist test set. The baseline_network achieved 74.10% accuracy and a capsule_dynamic model achieved 77.68% accuracy.


## Test 6 (mnist -> affnist)
**size of the input(in main.py) should be changed to 40x40 before running this test**  
Code to run the test
```
$ python main.py --model=capsule_dynamic --data=mnist --random_pos=True
```
```
$ python main.py --is_train=False --model=capsule_dynamic --data=affnist
```
***Trained randomly positioned 40x40 mnist images***
![Alt text](images/mnist40.jpg?raw=true "mnist40") 
***Tested 40x40 affnist images***
![Alt text](images/affnist.jpg?raw=true "affnist") 

| Model            | Parameters | Test Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    35.4M   |  61.70 % |
| capsule_dynamic  |    13.5M   |  75.89 % |


This test runs two models on 40x40 randomly placed mnist images. No affine transformation is applied other than translation and natural transformation seen in the standard mnist. This test is performed to test the robustness of the capsule network to affine transformations.

Comparing the baseline network with capsule network with dynamic routing shows similar result as mentioned in the CapsNet paper. The baseline network achieves 61.70% accuracy and the capsule network achieves 75.89% accuray.

## ToDo
* Implement Matrix Capsules with EM Routing 



