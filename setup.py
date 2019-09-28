import setuptools
import sys
import re

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
                     'opencv-python'
                 ],
                 description='AGNES - Flexible Reinforcement Learning Framework with PyTorch',
                 author='Rotinov Egor',
                 url='https://github.com/rotinov/AGNES',
                 version='0.0.3')

import pkg_resources
tf_pkg = None
for tf_pkg_name in ['torch']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'PyTorch needed, of version above 1.0.0'
from distutils.version import LooseVersion
assert LooseVersion(re.sub(r'-?rc\d+$', '', tf_pkg.version)) >= LooseVersion('1.0.0')