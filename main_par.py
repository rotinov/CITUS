import subprocess

subprocess.run(["mpiexec", '-n',  "2", 'python',  '-m', 'mpi4py', 'main_for_par.py'])
