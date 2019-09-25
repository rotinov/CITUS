import setuptools
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setuptools.setup(name='agnes',
                 packages=setuptools.find_packages(),
                 install_requires=[
                     'gym',
                     'scipy',
                     'gym[atari]',
                     'mpi4py',
                     'Tensorboard',
                     'cloudpickle',
                     'numpy',
                     'torch',
                     'opencv-python'
                 ],
                 description='AGNES - Flexible Reinforcement Learning Framework with PyTorch',
                 author='Rotinov Egor',
                 url='https://github.com/rotinov/AGNES',
                 version='0.0.3')
