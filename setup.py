"""Installation script."""
import blocks_mindlab
from setuptools import find_packages, setup

version = blocks_mindlab.__version__

setup(
    name='blocks_mindlab',
    version=version,
    description='Tools for build and launch experiments with Blocks',
    author='Mindlab Group',
    packages=find_packages(exclude=['tests']),
    install_requires=['blocks', 'blocks_extras'],
    extras_require={'test': ['mock', 'nose', 'nose2'], },
    scripts=['bin/run_all', 'bin/job'],
    zip_safe=False,
)
