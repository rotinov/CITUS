import subprocess

subprocess.run(["mpiexec", '-n',  "2", 'python',  '-m', 'mpi4py', 'examples/main_for_par.py'])
