# AGNES - Flexible Reinforcement Learning Framework with PyTorch

**Status:** This framework is under development and bugs may occur.

[![Build status](https://travis-ci.org/rotinov/AGNES.svg?branch=master)](https://travis-ci.org/rotinov/AGNES)

## Results
[![Peaking at 5362 at the end. The ending average is 5278.](results/MuJoCo-Ant-v2-PPO-1M/reward_per_update.svg?raw=true&sanitize=true)](results/MuJoCo-Ant-v2-PPO-1M)
*MuJoCo "Ant-v2" training with 1M steps. "Single" runner with "PPO" algorithm, MLP NN and 32 number of envs.*

*You can get the Tensorboard log file by clicking the image below(You will be redirected to the destination GitHub folder). The default config for the MuJoCo environment was used.*

## Runners
* Single

One worker and trainer. 'agnes.make_vec_env' can be used here.

```python
import agnes
import time


if __name__ == '__main__':
    env = agnes.make_env("InvertedDoublePendulum-v2")
    runner = agnes.Single(env, agnes.PPO, agnes.MLP)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"+str(time.time())))
    runner.run()

```

'agnes.log' - object of StandardLogger class that outputs parameters to console.
'agnes.TensorboardLogger' - class for writing logs in Tensorboard file.


* Distributed

Runs with

```bash
mpiexec -n 3 python -m mpi4py script_name.py
or
mpirun -n 3 python -m mpi4py script_name.py
```
This command will run 2 workers and 1 trainer.
```python
# script_name.py
import agnes


if __name__ == '__main__':
    env = agnes.make_vec_env("BreakoutNoFrameskip-v4")
    runner = agnes.Distributed(env, agnes.PPO, agnes.CNN)
    runner.run()

```

## Algorithms
* PPO
Proximal Policy Optimization is implemented in this framework and can be used simply:
```python
import agnes


if __name__ == '__main__':
    runner = agnes.Single(env, agnes.PPO, agnes.MLP)
    runner.run()

```

## Neural Network Architectures

* Multi Layer Perceptron

Can be used with both Continuous and Discrete action spaces.
```python
...
runner = agnes.Single(env, agnes.PPO, agnes.MLP)
...
```

* Convolutional Neural Network

Can be used only with Discrete action spaces.
```python
...
runner = agnes.Single(env, agnes.PPO, agnes.CNN)
...
```

* Recurrent Neural Network

Can be used only with Discrete action spaces.
```python
...
runner = agnes.Single(env, agnes.PPO, agnes.RNN)
...
```
