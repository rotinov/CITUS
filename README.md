# AGNES - Flexible Reinforcement Learning Framework with PyTorch

**Status:** This framework is under development and bugs may occur.

[![Build status](https://travis-ci.org/rotinov/AGNES.svg?branch=master)](https://travis-ci.org/rotinov/AGNES)

## Runners
* Single

```python
from agnes.nns import MLP
from agnes.algos import PPO
from agnes.runners import Single
import gym

env = gym.make("InvertedDoublePendulum-v2")
runner = Single(env, PPO, MLP)

```

* Distributed

```python

from agnes.nns import MLP
from agnes.algos import PPO
from agnes.runners import Distributed

runner = Distributed(env, PPO, MLP)

```

## Algorithms
* PPO
Proximal Policy Optimization is implemented in this framework and can be used simply:
```python
from agnes.nns import MLP
from agnes.algos import PPO
from agnes.runners import Single

runner = Single(env, PPO, MLP)

```

## Neural Network Architectures

* Multi Layer Perceptron
```python
from agnes.nns import MLP
from agnes.algos import PPO
from agnes.runners import Single

runner = Single(env, PPO, MLP)

```

* Convolutional Neural Network
```python
from agnes.nns import CNN
from agnes.algos import PPO
from agnes.runners import Single

runner = Single(env, PPO, CNN)

```
