module DualOptimalFiltering

using Base.Iterators: haskey
using Base.Iterators: include
using Base.Iterators: include

export dirichletkernel, filter_CIR, filter_WF, generate_CIR_trajectory


include("mcmc_sampler.jl")
include("common_utility_functions.jl")
include("pruning_functions.jl")
include("general_smoothing_functions.jl")
include("kde_for_pf_samples.jl")
include("dirichlet_kde.jl")
include("exact_L2_distances.jl")
include("post_process_Dirichlet_mixture_posterior.jl")

include("functions_CIR.jl")
include("filtering_CIR.jl")
include("approx_filtering_CIR.jl")
include("likelihood_CIR.jl")
include("smoothing_CIR.jl")
include("approx_smoothing_CIR.jl")
include("joint_smoothing_CIR.jl")
include("reparam_CIR.jl")
include("full_inference_CIR.jl")


include("functions_WF.jl")
include("filtering_WF.jl")
include("approx_filtering_WF.jl")
include("likelihood_WF.jl")
include("particle_filtering_WF.jl")

#include("exponentially_weighted_dirichlet.jl")
include("CIR_particle_approximations.jl")

#include("MC_approx_WF_selection.jl")
#include("MC_approx_WF_neutral.jl")

include("Neutral_WF_particle_approximations.jl")

end # module
