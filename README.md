# rust_tbc_parab_schwarz

This repository is a part of my MCS thesis, where the aim is to analyze transparent boundary conditions
(TBC) of a coupled system formed by 1D diffusions equations.
The diffusions equations may have a variable coefficient (in space) and non-uniform discretization.
This particular repository is a sub-module of a Python repository, at
    https://github.com/nuftau/schwarz_tbc_finder

There may not be a lot of documentation, because the functions are directly translated from their documented python equivalent.
In the same way, the module itself is not tested a lot because it can be directly tested against the python results.
See the Python repository to see comparison between results of python and rust.

This module was designed to be included in python code to compute the computationnaly-expensive task to integrate our model over time and extract the convergence rate of the Schwarz method. 
