import agnes.common.tests as test_pack


if __name__ == '__main__':
    test_pack.MLP_Discrete.test_single()
    test_pack.MLP_Continuous.test_single()
    test_pack.MLP_Continuous.test_vec()
    test_pack.CNN_Discrete.test_single()
    test_pack.RNN_Discrete.test_single()
    test_pack.MPI_runner.test_single()
