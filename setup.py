#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['scikit-plot',
                'tabulate',
                'pandas',
                'numpy',
                'h5py',
                'xgboost',
                'scikit-learn==0.20',
                'inquirer']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Yilun Guan",
    author_email='zoom.aaron@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Machine Learning Pipeline for ACT",
    entry_points={
        'console_scripts': [
            'mlpipe=mlpipe.cli:main',
        ],
    },
    install_requires=requirements,
    scripts=['bin/inspect_dataset',
             'bin/merge_datasets',
             'bin/generate_tod_list'],
    license="MIT license",
    include_package_data=True,
    keywords='mlpipe',
    name='mlpipe',
    packages=find_packages(include=['mlpipe']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/guanyilun/mlpipe',
    version='0.1.0',
    zip_safe=False,
)
