# AGNES - Flexible Reinforcement Learning Framework with PyTorch

**Status:** This framework is under active development and bugs may occur.

[![Build status](https://travis-ci.org/rotinov/AGNES.svg?branch=master)](https://travis-ci.org/rotinov/AGNES)

## Results
#### MuJoCo
**(Current results)**
[![The ending average is 5326.2](results/MuJoCo/reward_per_timestep.svg?raw=true&sanitize=true)](results/MuJoCo/Ant-v2_MLP)
*MuJoCo "Ant-v2" training with 1M steps. **Single** runner with **PPO** algorithm, **MLP** NN and 32 number of envs. The curve is an average of 3 runs.*

*You can get the Tensorboard log file by clicking the image above(You will be redirected to the destination GitHub folder). The default config for the MuJoCo environment was used. Plotted by **examples/plot.py***

#### Atari
**(Old results)**

[![Peaking at 861.8 at the end. The ending average is 854.8.](results/Atari-BreakoutNoFrameskip-v4-PPO-10M/reward_per_update.svg?raw=true&sanitize=true)](results/Atari-BreakoutNoFrameskip-v4-PPO-10M)
*Atari "BreakoutNoFrameskip-v4" with frame stack training with 10M steps. **DistributedMPI** runner with **PPO** algorithm, **LSTMCNN** and 16 number of envs.*

*You can get the Tensorboard log file by clicking the image above(You will be redirected to the destination GitHub folder). The default config for the Atari environment was used.*

![LSTMCNN agent plays Breakout](results/Atari-BreakoutNoFrameskip-v4-PPO-10M/Breakout-LSTMCNN.gif)

*Grad-cam technique was used for sampled action chosen by trained LSTMCNN(previous point).*

![LSTMCNN agent plays Breakout](results/Atari-BreakoutNoFrameskip-v4-PPO-10M/Breakout-LSTMCNN-Grad-Cam.gif)

## Runners
#### Single

One worker and trainer. **agnes.make_vec_env** can also be used here.

```python
import agnes
import time


if __name__ == '__main__':
    env = agnes.make_env("InvertedDoublePendulum-v2")
    runner = agnes.Single(env, agnes.PPO, agnes.MLP)
    runner.log(agnes.log, agnes.TensorboardLogger(".logs/"), agnes.CsvLogger(".logs/"))
    runner.run()

```

**agnes.log** - object of **StandardLogger** class that outputs parameters to console.
**agnes.TensorboardLogger** - class for writing logs in Tensorboard file.
**agnes.CsvLogger** - class for writing logs in csv file. (required for plotting)


#### DistributedMPI

Unlike in **Single** runner, in **DistributedMPI** runner due to async executing, weights are delayed by one rollout but this has no effect on learning because weights are delayed only by one update as it is in **Single** runner. So all parameters like probabilities ratio stay the same.

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
    runner = agnes.DistributedMPI(env, agnes.PPO, agnes.CNN)
    runner.run()

```

## Algorithms
#### A2C
Sync version of Advantage Actor Critic is implemented in this framework and can be used simply:
```python
import agnes


if __name__ == '__main__':
    runner = agnes.Single(env, agnes.A2C, agnes.MLP)
    runner.run()

```

#### PPO
Proximal Policy Optimization is implemented in this framework and can be used simply:
```python
import agnes


if __name__ == '__main__':
    runner = agnes.Single(env, agnes.PPO, agnes.MLP)
    runner.run()

```

## Neural Network Architectures

#### Multi Layer Perceptron

Can be used with both continuous and discrete action spaces.
```python
...
runner = agnes.Single(env, agnes.PPO, agnes.MLP)
...
```

#### Convolutional Neural Network

Can be used only with discrete action spaces.
```python
...
runner = agnes.Single(env, agnes.PPO, agnes.CNN)
...
```

#### Recurrent Neural Network

Can be used with both continuous and discrete action spaces.
```python
...
runner = agnes.Single(env, agnes.PPO, agnes.RNN)
...
```

#### Convolutional Recurrent Neural Network

Can be used only with discrete action spaces.
```python
...
runner = agnes.Single(env, agnes.PPO, agnes.RNNCNN)
...
```

#### Convolutional Neural Network with last LSTM layer
Can be used only with discrete action spaces.
```python
...
runner = agnes.Single(env, agnes.PPO, agnes.LSTMCNN)
...
```

## Make environment
* **make_vec_env(env, envs_num**=ncpu, **config**=None)**

    Parameters:
    * **env**(str or function) is id of gym environment or function, that returns initialized environment
    * **envs_num**(int) is a number of environments to initialize, by default is a number of logical cores on the CPU
    * **config**(dict) is a dictionary with parameters for **Monitor** and for initializing environment, by default is None(uses default config)

    Returns:
    * dict of
        1. **"env"**(**VecEnv** object)
        2. **"env_type"**(str)
        3. **"env_num"**(int) is a number of envs in **VecEnv** object
        4. **"env_name"**(str) is the name of envs in **VecEnv** object(Id in gym or class name)
    
    The whole tuple should be put in a **runner**.

* **make_env(env, config=None)** is an alias of **make_vec_env** without **envs_num** argument that will be setted to 1.

**Notice:** Some plot functions and environment wrappers were taken from [OpenAI Baselines(2017)](https://github.com/openai/baselines).
