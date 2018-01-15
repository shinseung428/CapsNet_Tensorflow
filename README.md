# Testing Capsule Network on various datasets

This repository contains different tests performed on a capsule network model. 

**Test 1 : Capsule Network on mnist dataset**  
**Test 2 : Capsule Network on fashion-mnist dataset**  
**Test 3 : Robustness of Capsule Network on randomly rotated mnist datset**  
**Test 4 : Robustness of Capsule Network on affine transformation**  


## Included datasets
* [mnist](http://yann.lecun.com/exdb/mnist/)
* [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)
* [affnist](http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/)
* smallNORB (Coming soon)

### path setting
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
      -extracted .mat files here
    -train
      -extracted .mat files here
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

| Model            | Parameters | Accuracy |
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

| Model            | Parameters | Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    13.2M   |  89.94%  |
| capsule_dynamic  |     8.2M   |  89.02%  |


Both baseline_network and capsule_dynamic network achieved about 89% accuracy on the fashion-mnist dataset. 


## Test 3 (mnist -> mnist(rotated))
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

| Model            | Parameters | Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    13.2M   |  74.10 % |
| capsule_dynamic  |     8.2M   |  77.68 % |


The baseline_network(CNN) and capsule_dynamic(CapsNet with dynamic routing) models were first trained on the normal 28x28 mnist dataset. Both network achieved high accuracy on the test set (both close to 99%). 

Two models were then tested on randomly rotated(-30 to +30) mnist test set. Figure 1 shows some of the inputs tested on both models. The baseline_network achieved ___ accuracy and a capsule_dynamic achieved ___ accuracy.


## Test 4 (mnist -> affnist)
**size of the input should be changed to 40x40 before running this code** 
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

| Model            | Parameters | Accuracy |
| ---------------- | ---------- | -------- |
| baseline_network |    35.4M   |  61.70 % |
| capsule_dynamic  |    13.5M   |  75.89 % |


This test runs two models on 40x40 randomly placed mnist images. No affine transformation is applied other than translation and natural transformation seen in the standard mnist. This test is performed to test the robustness of the capsule network to affine transformations.

Comparing the baseline network with capsule network with dynamic routing shows similar result as mentioned in CapsNet paper. The baseline network achieves 61.70% accuracy and the capsule network achieves 75.89% accuray.


