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
    version='0.0.3',
    long_description=README,
    url='https://github.com/lrcfmd/ipcsp',
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=['numpy', 'gurobipy', 'ase', 'numba', 'dwave-ocean-sdk', 'tabulate', 'pandas'],
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
    package_data={'ipcsp': ['data/CaAlSiO/*.lib', 'data/LiMgAlPO/*.lib',
                            'data/SrTiO/*.lib', 'data/YSrTiO/*.lib',
                            'data/ZnS/*.lib', 'data/ZnO/*.lib',
                            'data/Ewald/readme', 'data/grids/*.txt', 'data/grids/*.json',
                            'structures/*.cif', 'structures/spinel/*.*']}
)