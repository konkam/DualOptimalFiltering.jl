[![Coverage Status](https://coveralls.io/repos/github/konkam/DualOptimalFiltering.jl/badge.svg?branch=master)](https://coveralls.io/github/konkam/DualOptimalFiltering.jl?branch=master)

[![codecov](https://codecov.io/gh/konkam/DualOptimalFiltering.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/konkam/DualOptimalFiltering.jl)

[![Build Status](https://travis-ci.org/konkam/DualOptimalFiltering.jl.svg?branch=master)](https://travis-ci.org/konkam/DualOptimalFiltering.jl.svg?branch=master)

# DualOptimalFiltering

Optimal filtering, smoothing and general inference using a dual process.


This package provides a set of functions to perform exact optimal filtering, smoothing or general inference using a dual process.
We provide examples on the Cox-Ingersoll-Ross model with Poisson distributed data and the Wright-Fisher model with multinomial distributed data.

Plots are generated using `ggplot2` and require the installation of R, the R package `tidyverse` and the R-Julia interface RCall.jl
