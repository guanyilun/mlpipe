======
MLPipe
======


.. image:: https://img.shields.io/travis/guanyilun/mlpipe.svg
        :target: https://travis-ci.org/guanyilun/mlpipe

.. image:: https://readthedocs.org/projects/mlpipe/badge/?version=latest
        :target: https://mlpipe.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Machine Learning Pipeline for ACT


* Free software: MIT license
* Documentation: https://mlpipe.readthedocs.io.

Dependencies
--------

* pytorch
* h5py
* pandas
* tabulate
* numpy
* sklearn

Installation
--------

To install torch please follow the platform specific instructions on 
https://pytorch.org/

The rest of dependencies can be installed by

.. code-block:: bash

   pip install -r requirements.txt

The script assumes you have the required data file ``dataset.h5`` file located at ``data`` directory. On Feynman you can do the following, at the project root

.. code-block:: bash

   mkdir data
   cd data
   ln -s /mnt/act3/users/yilun/data/dataset.h5 .
   
Features
--------

* Train and test machine learning models in a uniform way
* Build-in weighted sampler to ensure training data are balanced
* Uniform metrics comparison of different models
* Automatically generate pandas report and save it for post process
* Expose GPU for easier GPU acceleration
* It replies on data from HDF5 file which is faster than reading from moby2

Sample outputs
-------
To run a test script, you can run the following

.. code-block:: bash
  
   python simple_pipe.py
   
This will produce the following output

.. code-block::

    epoch    batch  model           loss      base    accuracy    tp    tn    fp    fn    precision    recall        f1
  -------  -------  ----------  --------  --------  ----------  ----  ----  ----  ----  -----------  --------  --------
        0        0  CNNModel    19.9336   0.422877    0.422877  7172     0  9788     0     0.422877  1         0.594397
        0        0  KNNModel-7   1.74531  0.422877    0.949469  7169  8934   854     3     0.893556  0.999582  0.9436
        0        0  KNNModel-5   1.7616   0.422877    0.948998  7172  8923   865     0     0.892373  1         0.943126
        
    epoch    batch  model           loss      base    accuracy    tp    tn    fp    fn    precision    recall        f1
  -------  -------  ----------  --------  --------  ----------  ----  ----  ----  ----  -----------  --------  --------
        0      100  CNNModel    19.398    0.422877    0.438384  7163   272  9516     9     0.429462  0.998745  0.600646
        0      100  KNNModel-7   1.77992  0.422877    0.948467  7013  9073   715   159     0.907479  0.97783   0.941342
        0      100  KNNModel-5   1.63126  0.422877    0.952771  7146  9013   775    26     0.902159  0.996375  0.946929
        
    epoch    batch  model          loss      base    accuracy    tp    tn    fp    fn    precision    recall        f1
  -------  -------  ----------  -------  --------  ----------  ----  ----  ----  ----  -----------  --------  --------
        0      200  CNNModel    16.3369  0.422877    0.527005  3986  4952  4836  3186     0.451825  0.555772  0.498437
        0      200  KNNModel-7   1.7616  0.422877    0.948998  7172  8923   865     0     0.892373  1         0.943126
        0      200  KNNModel-5   1.7616  0.422877    0.948998  7172  8923   865     0     0.892373  1         0.943126
        
    epoch    batch  model          loss      base    accuracy    tp    tn    fp    fn    precision    recall        f1
  -------  -------  ----------  -------  --------  ----------  ----  ----  ----  ----  -----------  --------  --------
        0      300  CNNModel    16.1965  0.422877    0.531073  6119  2888  6900  1053     0.470005  0.853179  0.606112
        0      300  KNNModel-7   1.7616  0.422877    0.948998  7171  8924   864     1     0.89247   0.999861  0.943118
        0      300  KNNModel-5   1.7616  0.422877    0.948998  7172  8923   865     0     0.892373  1         0.943126
  
  == TEST RESULTS: ==

    epoch    batch  model           loss     base    accuracy    tp    tn    fp    fn    precision    recall        f1
  -------  -------  ----------  --------  -------  ----------  ----  ----  ----  ----  -----------  --------  --------
       -1        0  CNNModel    17.3614   0.43809    0.497347  5997  2438  7092  1433     0.458171  0.807133  0.584531
       -1        0  KNNModel-7   1.64143  0.43809    0.952476  7203  8951   579   227     0.925598  0.969448  0.947016
       -1        0  KNNModel-5   1.24025  0.43809    0.964092  7417  8934   596    13     0.925621  0.99825   0.960565   
     
Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
