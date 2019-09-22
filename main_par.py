import subprocess

subprocess.run(["mpiexec", '-n',  "3", 'python',  '-m', 'mpi4py', 'main_for_par.py'])
