using RCall, DataFrames, DataFramesMeta, JLD, DualOptimalFiltering, Distributions, Random, Weave, StatsFuns, FeynmanKacParticleFilters

include("filtering_error_helper_functions_CIR.jl")

x0 = 3 #Starting value
mean_ = 5
γ = 1.1
# sigma_grid = [0.5, 1., 2., 3.]
sigma_grid = [1.]
nparts_grid = [50, 100, 500, 1000]
nrep = 50

# Shorter run time for debugging
# sigma_grid = [0.5]
# nparts_grid = [50, 100]
# nrep = 5

induced_delta_grid = 2*γ*mean_ ./ sigma_grid.^2


to_plot = expand_grid(sigma = sigma_grid, Nparts = nparts_grid) |>
    df -> innerjoin(df, DataFrame(sigma = sigma_grid, delta = induced_delta_grid), on = :sigma) |>
    df -> estimate_errors_one_parameter_set.(df[!,:delta], 200, 20, x0, 0.5, df[!,:sigma], df[!,:Nparts], nrep)  |>
    dflist -> vcat(dflist...)

R"library(tidyverse)"

R"$to_plot %>%
    as_tibble %>%
    write_csv('saves_for_figures/filtering_error_varying_sigma.csv')"


