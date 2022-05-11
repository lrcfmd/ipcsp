"""Setup for the ipcsp package."""

import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Vladimir Gusev",
    author_email="vladimir.gusev@liverpool.ac.uk",
    name='ipcsp',
    license="MIT",
    description='Integer programming encoding for crystal structure prediction',
    version='v0.0.3',
    long_description=README,
    url='https://github.com/lrcfmd/ipcsp',
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=['numpy', 'gurobipy', 'ase', 'numba'],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
    ],
)