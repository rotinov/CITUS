from common.envs_prep.atari_wrappers import wrap_deepmind, make_atari

import gym

from common.envs_prep.subproc_vec_env import SubprocVecEnv

import multiprocessing


def wrap_vec_atari(env_name, envs_num=multiprocessing.cpu_count()):
    def make_env():
        def _thunk():
            env = wrap_deepmind(make_atari(env_name), frame_stack=True, clip_rewards=False)
            return env

        return _thunk

    envs = [make_env() for _ in range(envs_num)]
    envs = SubprocVecEnv(envs)

    env = make_env()()

    return envs, env, envs_num


def wrap_vec_gym(env_name, envs_num=multiprocessing.cpu_count()):
    def make_env():
        def _thunk():
            env = gym.make(env_name)
            return env

        return _thunk

    envs = [make_env() for _ in range(envs_num)]
    envs = SubprocVecEnv(envs)

    env = make_env()()

    return envs, env, envs_num
