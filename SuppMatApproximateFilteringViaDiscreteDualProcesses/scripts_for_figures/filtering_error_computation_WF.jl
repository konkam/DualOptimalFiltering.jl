using DataFrames, DataFramesMeta, JLD, DualOptimalFiltering, Distributions, Random, Weave, StatsFuns, FeynmanKacParticleFilters, ExactWrightFisher, CSV

include("filtering_error_helper_functions_WF.jl")

nparts_grid = [1000]
time_step_grid = [0.01, 0.5]
nrep = 100
nsteps = 16

induced_maxt_grid = time_step_grid*nsteps


α = repeat([1.1], inner = 3)

to_plot = expand_grid(maxt = induced_maxt_grid, Nparts = nparts_grid) |>
    df -> innerjoin(df, DataFrame(maxt = induced_maxt_grid, time_step = time_step_grid), on = :maxt) |>
    df -> @transform(df, :alpha = first(α), :K = length(α)) |>
    df -> estimate_errors_one_parameter_set.(Ref(α), nsteps, df[!,:maxt], Ref([0.4, 0.4, 0.2]), df[!,:Nparts], nrep)  |>
    dflist -> vcat(dflist...)


CSV.write("saves_for_figures/filtering_error_WF.csv", to_plot)

