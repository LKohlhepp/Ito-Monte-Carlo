# Ito-Monte-Carlo

Monte Carlo Simulation for simulating Ito's stochastic differential equations

Following: 
[Krülls, W. M.; Achterberg, A. Computation of cosmic-ray acceleration by Ito's stochastic differential equations. 
Astronomy and Astrophysics, Vol. 286, p. 314-327. 1994](https://ui.adsabs.harvard.edu/abs/1994A%26A...286..314K/abstract)

This is a particle shock simulation in AGN jets. By choosing different parameters in the simulations 
Fermi shocks of 1st and 2nd order as well as the syncrotron process can be simulated. 

The file ```sed_solvle.py``` contains the more general version that allows general functions for some of the parameters. 

The file ```sed_fast.py``` is more specific and is costumized for the parameter and function choices of Krülls (1994). 
Therefore, it could be programmed to utilize the GPU running mulitple Monte-Carlo-Simulations in parallel. 

Requirments:
* ```sed_solvle.py```:
  + numpy
* ```sed_fast.py```
  + numpy
  + numba
  + cudatoolkit
