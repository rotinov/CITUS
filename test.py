from agnes.common.tests import MLP_Discrete
from agnes.common.tests import MLP_Continuous
from agnes.common.tests import CNN_Discrete


if __name__ == '__main__':
    MLP_Discrete.test_single()
    MLP_Continuous.test_single()
    MLP_Continuous.test_vec()
    CNN_Discrete.test_single()
