using Test

using DualOptimalFiltering
# Run tests
@time @test 1 == 1

include("test_CIR_functions.jl")
include("test_mcmc_sampler.jl")
