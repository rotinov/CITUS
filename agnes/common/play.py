from agnes.algos.base import _BaseAlgo
import time
from collections import deque


class Visualize:
    def __init__(self, algo: _BaseAlgo, env):
        self.worker = algo
        self.env, _, _ = env

    def run(self):
        state = self.env.reset()
        self.worker.reset()
        self.env.render()
        done = [0]

        fps = 60

        smooth_delay = deque(maxlen=2*fps)

        while True:
            begin_time = time.perf_counter()
            action, pred_action, out = self.worker(state, done)
            nstate, reward, done, infos = self.env.step(action)
            self.env.render()
            now_time = time.perf_counter()
            smooth_delay.append(max(0., (1. / fps - (now_time - begin_time))))
            time.sleep(max(0., sum(smooth_delay) / len(smooth_delay)))
            state = nstate
