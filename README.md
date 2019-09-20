# CITUS - Flexible Reinforcement Learning Framework with PyTorch

This framework is under development and bugs may occur. 

# Runners
* Single

```
from runners import Single

runner = Single(env, PPO, MLP)
```

# Algorithms
* PPO
Proximal Policy Optimization is implemented in this framework and can be used simply:
```
from algos import PPO

runner = Single(env, PPO, MLP)
```

# Neural Network Architectures

* Multi Layer Perceptron
```
from nns import MLP

runner = Single(env, PPO, MLP)
```
