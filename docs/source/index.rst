.. mpca documentation master file, created by
   sphinx-quickstart on Thu Mar 21 16:39:39 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


mpca - methods to handle missing data in PCA
================================================================
mpca contains implementations of various methods to solve the following general problem:

Given a PCA model that has been defined on a train set **X**
and
a new sample **z**, with some variables missing:

estimate scores **t'** for **z** using the same PCA model
s.t. the difference **t'** - **t** is minimized

where **t** are the true scores of **z**
(true scores defined as the scores obtained from the PCA model when all data of **z** is observed)


Contents
==================

.. toctree::
    :maxdepth: 5
   Estimation
   Utilities


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
