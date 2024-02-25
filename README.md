# Numerically Stable Gaussian Processes

Gaussian Processes (GPs) are non-parametric methods for approximating the distribution over a set of functions directly. A gaussian process defines a prior over functions. After intaking observations gaussian processes can analytically solve for the posterior over the functions. GPs can be used for regression and classification.   

The focus of this repository is to build a set of GP examples that use computations that are numerically stable. As GPs utilize matrix multiplication, numerically instability can arise if computations are not handled carefully.

These examples are developed and inspired from this repository:
https://github.com/krasserm/bayesian-machine-learning
