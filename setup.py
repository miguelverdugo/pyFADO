# -*- coding: utf-8 -*-
"""
    Setup file for pyFADO.
"""

import setuptools


with open('README.md') as f:
    __readme__ = f.read()

with open('LICENSE') as f:
    __license__ = f.read()


def setup_package():
    setuptools.setup(
        name='pyFADO',
        version='0.0',
        description="Tool to extract information from FADO output files",
        long_description=__readme__,
        long_description_content_type='text/markdown',
        author='Miguel Verdugo',
        license="MIT",
        author_email='miguel.verdugo@univie.ac.at',
        url='https://github.com/miguelverdugo/pyFADO',
        package_dir={'pyfado': 'pyfado'},
        packages=['pyfado'],
        package_data={'pyfado': ['pyfado/data/*']},
        include_package_data=True,
        install_requires=['numpy',
                          'astropy',
                          'matplotlib',
                          'python-box',
                          'specutils'],
        classifiers=["Programming Language :: Python :: 3.7",
                        "License :: OSI Approved :: MIT License",
                        "Operating System :: OS Independent",
                        "Intended Audience :: Science/Research",
                        "Topic :: Scientific/Engineering :: Astronomy", ]
    )


if __name__ == "__main__":
    setup_package()


