from agnes.nns import mlp, cnn, rnn, base
from gym import spaces
from abc import ABC


class _BaseChooser(ABC):
    meta = "BASE"

    def __init__(self):
        pass

    def config(self, *args):
        return self

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space) -> base._BasePolicy:
        pass


class MLPChooser(_BaseChooser):
    meta = "MLP"

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
        if len(observation_space.shape) == 3:
            warnings.warn("Looks like you're using MLP for images. CNN is recommended.")

        if isinstance(action_space, spaces.Box):
            return mlp.MLPContinuous(observation_space, action_space)
        else:
            return mlp.MLPDiscrete(observation_space, action_space)


class CNNChooser(_BaseChooser):
    meta = "CNN"

    shared = True
    policy_nn = None
    value_nn = None
    nn = cnn.CNNDiscreteShared

    def config(self, shared=True, policy_nn=None, value_nn=None):
        if shared:
            if policy_nn is not None or value_nn is not None:
                raise NameError('Shared network with custom layers is not supported for now.')

            self.nn = cnn.CNNDiscreteShared
        else:
            self.nn = cnn.CNNDiscreteCopy
            self.policy_nn = policy_nn
            self.value_nn = value_nn
        return self

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
        if isinstance(action_space, spaces.Box):
            raise NameError('Continuous environments are not supported yet.')

        if self.nn == cnn.CNNDiscreteShared:
            return self.nn(observation_space, action_space)
        else:
            return self.nn(observation_space, action_space, self.policy_nn, self.value_nn)


class RNNinit(_BaseChooser):
    meta = "RNN"

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
        if isinstance(action_space, spaces.Box):
            return rnn.RNNContinuous(observation_space, action_space)
        else:
            return rnn.RNNDiscrete(observation_space, action_space)


class RNNCNNinitializer(_BaseChooser):
    meta = "RNN-CNN"
    gru = False

    def config(self, gru):
        self.gru = gru
        return self

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
        return rnn.RNNCNNDiscrete(observation_space, action_space, gru=self.gru)


class LSTMCNNinitializer(_BaseChooser):
    meta = "LSTM-CNN"

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
        return rnn.LSTMCNNDiscrete(observation_space, action_space)


MLP = MLPChooser()

CNN = CNNChooser()

RNN = RNNinit()
RNNCNN = RNNCNNinitializer().config(gru=False)
GRUCNN = RNNCNNinitializer().config(gru=True)
LSTMCNN = LSTMCNNinitializer()
