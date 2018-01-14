# Testing Capsule Network on various datasets

## Included datasets
* mnist
* fashion-mnist
* affnist

## Network Model
* baseline_network (Convolutional Neural Network)
* capsule_dynamic (Capsule Network with Dynamic Routing)
* *capsule_em* (ToDo)

## Training
'''
$ python main.py --model=capsule_dynamic --data=mnist
'''

## Testing
'''
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist
'''

### Train: mnist     Test: mnist (random rotation)

### Train: fashion-mnist     Test: fashion-mnist

### Train: mnist (randomly placed mnist on 40x40 background)     Test: affnist

