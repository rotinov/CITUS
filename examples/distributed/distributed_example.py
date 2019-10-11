import subprocess


subprocess.run(["mpiexec", '-n',  "3",
                'python',  '-m', 'mpi4py',
                'examples/distributed/distributed_example_supplementary.py'])
