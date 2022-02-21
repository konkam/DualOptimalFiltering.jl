[![Coverage Status](https://coveralls.io/repos/github/konkam/DualOptimalFiltering.jl/badge.svg?branch=master)](https://coveralls.io/github/konkam/DualOptimalFiltering.jl?branch=master)
[![codecov](https://codecov.io/gh/konkam/DualOptimalFiltering.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/konkam/DualOptimalFiltering.jl)
[![Build Status](https://travis-ci.org/konkam/DualOptimalFiltering.jl.svg?branch=master)](https://travis-ci.org/konkam/DualOptimalFiltering.jl.svg?branch=master)

# DualOptimalFiltering

Optimal filtering, smoothing and general inference using a dual process.


This package provides a set of functions to perform exact optimal filtering, smoothing or general inference using a dual process.
We provide examples on the Cox-Ingersoll-Ross model with Poisson distributed data and the Wright-Fisher model with multinomial distributed data.

## Installation
````julia

import Pkg
Pkg.add(url="https://github.com/konkam/DualOptimalFiltering.jl")
````



## Usage

### Cox-Ingersoll-Ross Filtering example

#### Simulate some data

````julia
using DualOptimalFiltering, Random, Distributions

function simulate_CIR_data(;Nsteps_CIR = 50, Nobs = 5, δ = 3., γ = 2.5, σ = 4.)
    Random.seed!(1)

    dt_CIR = 0.011
    λ = 1.

    time_grid_CIR = [k*dt_CIR for k in 0:(Nsteps_CIR-1)]
    X_CIR = generate_CIR_trajectory(time_grid_CIR, 3, δ, γ, σ)
    Y_CIR = map(λ -> rand(Poisson(λ), Nobs), X_CIR);
    data_CIR = Dict(zip(time_grid_CIR, Y_CIR))
    return data_CIR, X_CIR,  δ, γ, σ, λ
end

data_CIR, X_CIR,  δ, γ, σ, λ = simulate_CIR_data()
````


````
(Dict(0.495 => [4, 9, 7, 5, 5],0.517 => [2, 2, 0, 4, 7],0.011 => [4, 5, 4, 
3, 4],0.32999999999999996 => [3, 5, 2, 2, 3],0.297 => [0, 2, 0, 2, 1],0.484
 => [8, 10, 6, 5, 5],0.022 => [1, 1, 2, 2, 1],0.418 => [0, 0, 2, 4, 1],0.34
099999999999997 => [3, 3, 2, 2, 0],0.528 => [6, 1, 8, 6, 7]…), [3.0, 2.7564
965319331467, 2.3880096847200876, 3.921167439497704, 9.30930170923265, 14.5
14888921347417, 14.993362852522825, 17.192544278532573, 15.658780461278619,
 9.335794822551398  …  5.79641237544099, 4.361014445690017, 5.2661303583857
86, 4.366796112276612, 5.470119203373088, 4.093767074651879, 6.463104909420
805, 3.6944492239067688, 4.69977297524652, 3.7518228017252353], 3.0, 2.5, 4
.0, 1.0)
````




#### Filter the data

````julia
Λ_of_t_CIR, wms_of_t_CIR, θ_of_t_CIR = filter_CIR(δ, γ, σ, λ, data_CIR; silence = true);
````





#### Plot the filtering distribution

Filtering distribution, 95% credible band and true hidden signal:


````julia
using Plots

function plot_data_and_posterior_distribution2(δ, θ_of_t, Λ_of_t, wms_of_t, data, X_CIR)
    times = keys(data) |> collect |> sort;
    psi_t = [DualOptimalFiltering.create_Gamma_mixture_density(δ, θ_of_t[t], Λ_of_t[t], wms_of_t[t]) for t in keys(data) |> collect |> sort];
    expect_mixture = [sum(wms_of_t[t].*(δ/2 .+ Λ_of_t[t]) ./ θ_of_t[t]) for t in times]
    qt0025 = [DualOptimalFiltering.compute_quantile_mixture_hpi(δ, θ_of_t[t], Λ_of_t[t], wms_of_t[t], 0.025) for t in keys(data) |> collect |> sort];
    qt0975 = [DualOptimalFiltering.compute_quantile_mixture_hpi(δ, θ_of_t[t], Λ_of_t[t], wms_of_t[t], 0.975) for t in keys(data) |> collect |> sort];

    y = range(0, stop = maximum(data |> values |> collect |> x -> vcat(x...)), length = 200)
    z = [f.(y) for f in psi_t] |> x -> hcat(x...)

    heatmap(times, y, z, color = :heat)

    plot!(times, [data[t] for t in times] |> X -> hcat(X...)', seriestype=:scatter, c=:black, legend = false)
    plot!(times, X_CIR, c=:blue)
    plot!(times, qt0025, c=:red)
    plot!(times, qt0975, c=:red)

end

pl = plot_data_and_posterior_distribution2(δ, θ_of_t_CIR, Λ_of_t_CIR, wms_of_t_CIR, data_CIR, X_CIR)
````


![](figures/README_4_1.png)
