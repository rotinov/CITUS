class BaseBuffer(object):
    def append(self, transition):
        pass

    def rollout(self):
        pass

    def learn(self, data, nminibatches):
        pass

    def __len__(self):
        pass


class BaseAlgo:
    def __call__(self, state):
        pass

    def experience(self, transition, timestep_now):
        pass

    def learn(self, input, timestep_now):
        pass
