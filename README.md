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

## Training
```
$ python main.py --model=capsule_dynamic --data=mnist
```
Code to train the network. Different models can be trained on various datasets. 

## Testing
```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist
```
Code to test the network. The model can be trained on one dataset and can be tested on another dataset. 

### Test 1
![Alt text](images/mnist_gt.jpg?raw=true "mnist")
![Alt text](images/mnist_rotated.jpg?raw=true "rotated mnist")

| Train | Test |
| ------ | ------------- |
| mnist  | mnist (random rotation -30 to +30) |


```
$ python main.py --model=capsule_dynamic --data=mnist
```
```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist --rotate=True
```
The baseline_network(CNN) and capsule_dynamic(CapsNet with dynamic routing) models were first trained on the normal 28x28 mnist dataset. Both network achieved high accuracy on the test set (both close to 99%). 

Two models were then tested on randomly rotated(-30 to +30) mnist test set. Figure 1 shows some of the inputs tested on both models. The baseline_network achieved ___ accuracy and a capsule_dynamic achieved ___ accuracy.

### Test 2 
![Alt text](images/fashion_mnist_gt.jpg?raw=true "fashion-mnist")
![Alt text](images/fashion_mnist_recon.jpg?raw=true "fashion-mnist reconstructed")
| Train | Test |
| -------------- | ------------- |
| fashion-mnist  | fashion-mnist |

```
$ python main.py --model=capsule_dynamic --data=fashion-mnist 
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=fashion-mnist 
```
The normal convolutional network achieved test accuracy of 91.17% while capsule network with dynamic routing achieved accuracy of 89.52%. 

#### Reconstructed fashion-mnist image


### Test 3
| Train | Test |
| ----- | ---- |
| mnist (randomly placed mnist on 40x40 background) | affnist |

```
$ python main.py --model=capsule_dynamic --data=mnist --random_pos=True
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=affnist
```
First code runs capsule_dynamic model on 40x40 randomly placed mnist image. Training images contain randomly placed mnist images. No affine transformation is applied other than translation and natural transformation seen in the standard mnist. 

Comparing the baseline network with capsule network with dynamic routing shows similar result as mentioned in CapsNet paper. Baseline network achieves accuracy of ___ and the capsule network achieves an accuray of ____.

#### Reconstructed randomly placed mnist image

