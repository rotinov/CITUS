# AGNES - Flexible Reinforcement Learning Framework with PyTorch

**Status:** This framework is under development and bugs may occur.

[![Build status](https://travis-ci.org/rotinov/AGNES.svg?branch=master)](https://travis-ci.org/rotinov/AGNES)

## Runners
* Single

```python
import agnes


env = agnes.make_env("InvertedDoublePendulum-v2")
runner = agnes.Single(env, agnes.PPO, agnes.MLP)

```

* Distributed

```python

import agnes


runner = agnes.Distributed(env, agnes.PPO, agnes.MLP)

```

## Algorithms
* PPO
Proximal Policy Optimization is implemented in this framework and can be used simply:
```python
import agnes


runner = agnes.Single(env, agnes.PPO, agnes.MLP)

```

## Neural Network Architectures

* Multi Layer Perceptron
```python
import agnes


runner = agnes.Single(env, agnes.PPO, agnes.MLP)

```

* Convolutional Neural Network
```python
import agnes


runner = agnes.Single(env, agnes.PPO, agnes.CNN)

```
