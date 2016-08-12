"""Installation script."""
from setuptools import find_packages, setup

setup(
    name='blocks_mindlab',
    description='Tools for build and launch experiments with Blocks',
    author='Mindlab Group',
    packages=find_packages(),
    install_requires=['blocks', 'blocks_extras'],
    extras_require={},
    scripts=['bin/run_all', 'bin/job'],
    zip_safe=False,
)
