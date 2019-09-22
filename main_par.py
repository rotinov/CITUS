import subprocess

subprocess.run(["mpiexec", '-n',  "16", 'python',  '-m', 'mpi4py', 'main_for_par.py'])
