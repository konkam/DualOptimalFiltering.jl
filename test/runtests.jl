using Test

using DualOptimalFiltering
# Run tests
@time @test 1 == 1

include("test_common_utility_functions.jl")
include("test_mcmc_sampler.jl")
include("test_pruning_functions.jl")

include("test_CIR_functions.jl")
include("test_filtering_CIR.jl")
include("test_approx_filtering_CIR.jl")
include("test_likelihood_CIR.jl")
