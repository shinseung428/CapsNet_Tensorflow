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
```
$ python main.py --model=capsule_dynamic --data=mnist
```

## Testing
```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist
```

### Train: mnist     Test: mnist (random rotation)
```
$ python main.py --model=capsule_dynamic --data=mnist
```
```
$ python main.py --is_train=False --model=capsule_dynamic --data=mnist --rotate=True
```

### Train: fashion-mnist     Test: fashion-mnist
```
$ python main.py --model=capsule_dynamic --data=fashion-mnist 
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=fashion-mnist 
```

### Train: mnist (randomly placed mnist on 40x40 background)     Test: affnist
```
$ python main.py --model=capsule_dynamic --data=mnist --random_pos=True
```

```
$ python main.py --is_train=False --model=capsule_dynamic --data=affnist
```
