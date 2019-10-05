import agnes


class Visualize:
    def __init__(self, algo, env):
        self.worker = algo
        self.env, _, _ = env

    def run(self):
        state = self.env.reset()
        self.env.render()

        while True:
            action, pred_action, out = self.worker(state)
            nstate, reward, done, infos = self.env.step(action)
            self.env.render()
            state = nstate
