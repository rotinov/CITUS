import gym
import re
from collections import defaultdict
import multiprocessing

from agnes.common.envs_prep import wrap_deepmind, make_atari, SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from agnes.common.envs_prep import Monitor


_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


def make_vec_env(env, envs_num=multiprocessing.cpu_count()):
    if isinstance(env, str):
        env_type, env_id = get_env_type(env)

        if env_type == 'atari':
            envs, num_envs = wrap_vec_atari(env_id, envs_num=envs_num)
        else:
            envs, num_envs = wrap_vec_gym(env_id, envs_num=envs_num)
    else:
        envs, num_envs = wrap_vec_custom(env, envs_num=envs_num)
        env_type = 'custom'

    if env_type == 'mujoco':
        envs = VecNormalize(envs)

    return envs, env_type, num_envs


def make_env(env):
    if isinstance(env, str):
        env_type, env_id = get_env_type(env)

        if env_type == 'atari':
            envs, num_envs = wrap_vec_atari(env_id, envs_num=1)
        else:
            envs, num_envs = wrap_vec_gym(env_id, envs_num=1)
    else:
        envs, num_envs = wrap_vec_custom(env, envs_num=1)
        env_type = 'custom'

    if env_type == 'mujoco':
        envs = VecNormalize(envs)

    return envs, env_type, 1


def get_env_type(env: str):
    env_id = env

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types {}'.format(env_id, _game_envs.keys())

    return env_type, env_id


def wrap_vec_atari(env_name, envs_num=multiprocessing.cpu_count()):
    def make_env():
        def _thunk():
            env = Monitor(wrap_deepmind(make_atari(env_name)), allow_early_resets=True)
            return env

        return _thunk

    envs = [make_env() for _ in range(envs_num)]

    if envs_num == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    envs = VecFrameStack(envs, nstack=4)

    return envs, envs_num


def wrap_vec_gym(env_name, envs_num=multiprocessing.cpu_count()):
    def make_env():
        def _thunk():
            return Monitor(gym.make(env_name), allow_early_resets=True)

        return _thunk

    envs = [make_env() for _ in range(envs_num)]

    if envs_num == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    return envs, envs_num


def wrap_vec_custom(env_init_fun, envs_num=multiprocessing.cpu_count()):
    def make_env():
        return Monitor(env_init_fun, allow_early_resets=True)

    envs = [make_env() for _ in range(envs_num)]

    if envs_num == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    return envs, envs_num
