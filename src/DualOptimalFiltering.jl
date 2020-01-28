module DualOptimalFiltering

include("mcmc_sampler.jl")
include("common_utility_functions.jl")
include("pruning_functions.jl")
include("general_smoothing_functions.jl")

include("CIR_functions.jl")
include("filtering_CIR.jl")
include("approx_filtering_CIR.jl")
include("likelihood_CIR.jl")
include("smoothing_CIR.jl")
end # module
