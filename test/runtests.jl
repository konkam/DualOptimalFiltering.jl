using Test

using DualOptimalFiltering
# Run tests
@time @test 1 == 1

include("test_common_utility_functions.jl")
include("test_mcmc_sampler.jl")
include("test_pruning_functions.jl")
include("test_kde_for_pf_samples.jl")
include("test_dirichlet_kde.jl")


include("test_functions_CIR.jl")
include("test_filtering_CIR.jl")
include("test_approx_filtering_CIR.jl")
include("test_likelihood_CIR.jl")
include("test_smoothing_CIR.jl")
include("test_joint_smoothing_CIR.jl")

include("test_approx_filtering_WF.jl")
include("test_likelihood_WF.jl")
include("test_particle_filtering_WF.jl")
