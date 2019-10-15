import subprocess


subprocess.run(["mpiexec", '-n',  "3",
                'python',  '-m', 'mpi4py',
                'examples/distributed_rnn/distributed_example_supplementary.py'])
