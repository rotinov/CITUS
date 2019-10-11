from agnes.nns import mlp, cnn, rnn
from gym import spaces
from abc import ABC


class _BaseChooser(ABC):
    def __init__(self):
        pass

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
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

    def __init__(self, shared=True, policy_nn=None, value_nn=None):
        super().__init__()

        if shared:
            if policy_nn is not None or value_nn is not None:
                raise NameError('Shared network with custom layers is not supported for now.')

            self.nn = cnn.CNNDiscreteShared
        else:
            self.nn = cnn.CNNDiscreteCopy
            self.policy_nn = policy_nn
            self.value_nn = value_nn

    def __call__(self, observation_space, action_space):
        if isinstance(action_space, spaces.Box):
            raise NameError('Continuous environments are not supported yet.')

        if self.nn == cnn.CNNDiscreteShared:
            return self.nn(observation_space, action_space)
        else:
            return self.nn(observation_space, action_space, self.policy_nn, self.value_nn)


class RNNinit(_BaseChooser):
    meta = "RNN"

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
        return rnn.RNNDiscrete(observation_space, action_space)


class RNNCNNinitializer(_BaseChooser):
    meta = "RNN-CNN"

    def __init__(self, gru=False):
        super().__init__()
        self.gru = gru

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
        return rnn.RNNCNNDiscrete(observation_space, action_space, gru=self.gru)


class LSTMCNNinitializer(_BaseChooser):
    meta = "LSTM-CNN"

    def __call__(self, observation_space: spaces.Space, action_space: spaces.Space):
        return rnn.LSTMCNNDiscrete(observation_space, action_space)


MLP = MLPChooser()

CNN = CNNChooser()

RNN = RNNinit()
RNNCNN = RNNCNNinitializer(gru=False)
GRUCNN = RNNCNNinitializer(gru=True)
LSTMCNN = LSTMCNNinitializer()
