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

* python2.7 (I am still working on python 3 compatiblity)
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

The script assumes you have the required data file ``dataset.h5`` located at the ``data`` directory. On Feynman you can do the following - at the project root

.. code-block:: bash

   mkdir data
   cd data
   ln -s /mnt/act3/users/yilun/data/dataset.h5 .
   
Features
--------

* Train and test machine learning models in a uniform way
* Handles random sampling of detector data and pickle parameters
* Build-in weighted sampler to ensure training data are balanced
* Uniform metrics comparison of different models
* Automatically generate pandas report and save it for post process
* Expose GPU for easier GPU acceleration
* Read data from an HDF5 file which is faster than from moby2 and more scalable

Sample outputs
-------
To run a test script, you can run the following

.. code-block:: bash
  
   python simple_pipe.py
   
This will produce the following output

.. code-block::

  == VALIDATION RESULTS: ==

    epoch    batch  model               loss      base    accuracy    tp    tn    fp    fn    precision    recall        f1
  -------  -------  ---------------  -------  --------  ----------  ----  ----  ----  ----  -----------  --------  --------
        0        0  KNNModel-3       1.85934  0.422877    0.946167  6951  9096   692   221     0.90946   0.969186  0.938373
        0        0  KNNModel-7       1.62719  0.422877    0.952889  7103  9058   730    69     0.906805  0.990379  0.946751
        0        0  RandomForest-5   1.28301  0.422877    0.962854  7164  9166   622     8     0.920113  0.998885  0.957882
        0        0  KNNModel-5       1.70457  0.422877    0.950649  7049  9074   714   123     0.908025  0.98285   0.943957
        0        0  DecisionTree     1.87563  0.422877    0.945696  6798  9241   547   374     0.925528  0.947853  0.936557
        0        0  RandomForest-20  1.27894  0.422877    0.962972  7165  9167   621     7     0.920241  0.999024  0.958016
        0        0  SVCModel         1.87361  0.422877    0.945755  7093  8947   841    79     0.894001  0.988985  0.939097
        0        0  RandomForest-10  1.28301  0.422877    0.962854  7164  9166   622     8     0.920113  0.998885  0.957882

  == TEST RESULTS: ==

    epoch    batch  model                loss     base    accuracy    tp    tn    fp    fn    precision    recall        f1
  -------  -------  ---------------  --------  -------  ----------  ----  ----  ----  ----  -----------  --------  --------
       -1        0  KNNModel-3       1.43167   0.43809    0.95855   7153  9104   426   277     0.943792  0.962719  0.953161
       -1        0  KNNModel-7       1.12416   0.43809    0.967453  7327  9081   449   103     0.942258  0.986137  0.963699
       -1        0  RandomForest-5   0.684274  0.43809    0.980189  7427  9197   333     3     0.957088  0.999596  0.97788
       -1        0  KNNModel-5       1.2545    0.43809    0.963679  7251  9093   437   179     0.943158  0.975908  0.959254
       -1        0  DecisionTree     1.39704   0.43809    0.959552  7013  9261   269   417     0.96306   0.943876  0.953371
       -1        0  RandomForest-20  0.688348  0.43809    0.980071  7427  9195   335     3     0.956841  0.999596  0.977751
       -1        0  SVCModel         1.42557   0.43809    0.958726  7319  8941   589   111     0.925518  0.985061  0.954362
       -1        0  RandomForest-10  0.686311  0.43809    0.98013   7426  9197   333     4     0.957082  0.999462  0.977813
     
Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
