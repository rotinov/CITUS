import subprocess


def test_single():
    subprocess.run(["mpiexec", '-n',  "2", 'python',  '-m', 'mpi4py', 'agnes/common/tests/MPI_MLP_Discrete.py'])
