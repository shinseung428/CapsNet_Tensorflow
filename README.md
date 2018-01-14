# Testing Capsule Network on various datasets

## Included datasets
* mnist
* fashion-mnist
* affnist
* smallNORB (ToDo)

## Network Model
* baseline_network (Convolutional Neural Network)
* capsule_dynamic (Capsule Network with Dynamic Routing)
* *capsule_em* (ToDo)

## How to run Training & Testing
```
$ python main.py --model=capsule_dynamic --data=mnist
```
An example code to train the capsule_dynamic(CapsNet with dynamic routing) model on mnist dataset.

```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist
```
An example code to test the capsule_dynamic(CapsNet with dynamic routing) model on mnist dataset.

Other models can be trained/tested by changing the name of the --model flag, and other datasets can be used by changing the name of the --data flag.

## Test 1 (mnist -> mnist)
Code to run the test
```
$ python main.py --model=capsule_dynamic --data=mnist 
```
```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist 
```

![Alt text](images/mnist_gt.jpg?raw=true "mnist")
*** Trained mnist images ***
![Alt text](images/mnist_recon.jpg?raw=true "mnist reconstructed")
*** Reconstructed mnist images ***

| Model            | Accuracy |
| ---------------- | -------- |
| baseline_network |  99.03 % |
| capsule_dynamic  |  99.25 % |

Both baseline_network and capsule_dynamic network achieved above 99% accuracy on the mnist dataset. 

## Test 2 (fashoin -> mnist)

Code to run the test
```
$ python main.py --model=capsule_dynamic --data=fashion-mnist 
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=fashion-mnist 
```

*** Trained fashion-mnist images ***
![Alt text](images/fashion_mnist_gt.jpg?raw=true "fashion-mnist")
*** Reconstructed fashion-mnist images ***
![Alt text](images/fashion_mnist_recon.jpg?raw=true "fashion-mnist reconstructed") 

| Model            | Accuracy |
| ---------------- | -------- |
| baseline_network |  89.94%  |
| capsule_dynamic  |  89.02%  |


Both baseline_network and capsule_dynamic network achieved about 89% accuracy on the fashion-mnist dataset. 


## Test 3 (mnist -> mnist(rotated))
Code to run the test
```
$ python main.py --model=capsule_dynamic --data=mnist
```
```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist --rotate=True
```

*** Trained mnist images ***
![Alt text](images/mnist_gt.jpg?raw=true "mnist") 
*** Tested randomly rotated mnist images ***
![Alt text](images/mnist_rotated.jpg?raw=true "rotated mnist")

| Model            | Accuracy |
| ---------------- | -------- |
| baseline_network |  74.10 % |
| capsule_dynamic  |  77.68 % |


The baseline_network(CNN) and capsule_dynamic(CapsNet with dynamic routing) models were first trained on the normal 28x28 mnist dataset. Both network achieved high accuracy on the test set (both close to 99%). 

Two models were then tested on randomly rotated(-30 to +30) mnist test set. Figure 1 shows some of the inputs tested on both models. The baseline_network achieved ___ accuracy and a capsule_dynamic achieved ___ accuracy.


## Test 4 (mnist -> affnist)
Code to run the test
```
$ python main.py --model=capsule_dynamic --data=mnist --random_pos=True
```
```
$ python main.py --is_train=False --model=capsule_dynamic --data=affnist
```
*** Trained randomly positioned 40x40 mnist images ***
![Alt text](images/mnist40.jpg?raw=true "mnist40") 
*** Tested 40x40 affnist images ***
![Alt text](images/affnist.jpg?raw=true "affnist") 

| Model            | Accuracy |
| ---------------- | -------- |
| baseline_network |  61.70 % |
| capsule_dynamic  |  75.89 % |


This test runs two models on 40x40 randomly placed mnist images. No affine transformation is applied other than translation and natural transformation seen in the standard mnist. This test is performed to test the robustness of the capsule network to affine transformations.

Comparing the baseline network with capsule network with dynamic routing shows similar result as mentioned in CapsNet paper. The baseline network achieves 61.70% accuracy and the capsule network achieves 75.89% accuray.


