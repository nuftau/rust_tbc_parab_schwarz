# rust_tbc_parab_schwarz

This repository is a part of my PhD thesis, where the aim is to analyze Schwarz methods at a discrete levels
for a coupled system formed by 1D reaction-diffusions equations.
The diffusions equations may have a variable coefficient (in space) and non-uniform discretization.
This particular repository was a sub-module of a Python repository, at
    https://github.com/simon-clement/schwarz_discrete_convergence

There may not be a lot of documentation, because the functions are directly translated from their documented python equivalent.

This module was designed to be included in python code to compute the computationnaly-expensive task to integrate our model over time and extract the convergence rate of the Schwarz method. It was designed to solve only tridiagonal systems. Some discretizations are hence impossible to accelerate

Warning: some of the results obtained by this module can differ from python module. For instance, the schwarz method of finite difference with naive discretization of neumann condition has a strange behaviour. However, all the figures can be exported with python (the computational cost is then multiplied by 10).
For this reason, this repo was not used in the PhD manuscript.
