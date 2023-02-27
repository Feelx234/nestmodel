from setuptools import setup

setup(
    name='nestmodel',
    version='0.1.0',
    packages=['nestmodel', 'nestmodel.tests'],
    author='Felix Stamm',
    author_email='felix.stamm@cssh.rwth-aachen.de',
    description='This package contains the implementation of the neighborhood structure configuration model for python',
    install_requires=[
              'pandas', 'scipy', 'numpy', 'matplotlib', 'numba', 'networkx', 'brokenaxes', 'nbconvert', 'tqdm'
          ],
    python_requires='>=3.8'
)
