from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nestmodel',
    version='0.1.1',
    packages=['nestmodel', 'nestmodel.tests'],
    author='Felix Stamm',
    author_email='felix.stamm@cssh.rwth-aachen.de',
    description='This package contains the implementation of the neighborhood structure configuration model for python',
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    install_requires=[
              'pandas', 'scipy', 'numpy', 'matplotlib', 'numba', 'networkx', 'brokenaxes', 'nbconvert', 'tqdm'
          ],
    python_requires='>=3.8'
)
