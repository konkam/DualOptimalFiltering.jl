module DualOptimalFiltering

include("mcmc_sampler.jl")
include("common_utility_functions.jl")
include("pruning_functions.jl")
include("general_smoothing_functions.jl")

include("functions_CIR.jl")
include("filtering_CIR.jl")
include("approx_filtering_CIR.jl")
include("likelihood_CIR.jl")
include("smoothing_CIR.jl")
include("joint_smoothing_CIR.jl")

include("functions_WF.jl")
include("filtering_WF.jl")
include("approx_filtering_WF.jl")
include("likelihood_WF.jl")
end # module
