# AGNES - Flexible Reinforcement Learning Framework with PyTorch

**This framework is under development and bugs may occur.**

## Runners
* Single

```python
from runners import Single

runner = Single(env, PPO, MLP)
```

* Distributed

```python
from runners import Distributed

runner = Distributed(env, PPO, MLP)
```

## Algorithms
* PPO
Proximal Policy Optimization is implemented in this framework and can be used simply:
```python
from algos import PPO

runner = Single(env, PPO, MLP)
```

## Neural Network Architectures

* Multi Layer Perceptron
```python
from nns import MLP

runner = Single(env, PPO, MLP)
```

* Convolutional Neural Network
```python
from nns import CNN

runner = Single(env, PPO, CNN)
```
