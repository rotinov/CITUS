import agnes
from agnes.algos.base import BaseAlgo


class Visualize:
    def __init__(self, algo: BaseAlgo, env):
        self.worker = algo
        self.env, _, _ = env

    def run(self):
        state = self.env.reset()
        self.worker.reset()
        self.env.render()
        done = [0]

        while True:
            action, pred_action, out = self.worker(state, done)
            nstate, reward, done, infos = self.env.step(action)
            self.env.render()
            state = nstate
