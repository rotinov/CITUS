import algos
import nns
import runners
from common.envs_prep.atari_wrappers import wrap_deepmind, make_atari
from common.envs_prep.subproc_vec_env import SubprocVecEnv


env_name = "EnduroNoFrameskip-v4"


def make_env():
    def _thunk():
        return wrap_deepmind(make_atari(env_name), frame_stack=True, clip_rewards=False)

    return _thunk


if __name__ == '__main__':
    # env = gym.make("InvertedDoublePendulum-v2")
    env = wrap_deepmind(make_atari(env_name), frame_stack=True, clip_rewards=False)

    num_envs = 8
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    runner = runners.SingleVec(envs, algos.PPO, nns.CNN, workers_num=num_envs, all_cuda=True)
    runner.run()
