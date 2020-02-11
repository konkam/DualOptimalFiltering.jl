function log_cost_to_go_CIR_keep_fixed_number(δ, γ, σ, λ, data, fixed_number::Int64; silence = false)
    # println("filter_WF_mem2")

    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        return keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
    end

    Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t = compute_all_log_cost_to_go_functions_CIR_pruning(δ, γ, σ, λ, data, prune_keeping_fixed_number; silence = silence)

    return Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t
end

function smooth_CIR_keep_fixed_number(δ, γ, σ, λ, data, fixed_number::Int64; silence = false)
    # println("filter_WF_mem2")
    α = δ/2
    β = γ/σ^2

    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    Λ_of_t, logwms_of_t, θ_of_t = filter_CIR_pruning_logweights(δ, γ, σ, λ, data, prune_keeping_fixed_number; silence = silence)
    Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t = log_cost_to_go_CIR_keep_fixed_number(δ, γ, σ, λ, data, fixed_number::Int64; silence = silence)

    Λ_of_t_smooth, wms_of_t_smooth, θ_of_t_smooth = merge_filtering_and_cost_to_go_logweights_CIR(Λ_of_t, logwms_of_t, θ_of_t, Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t, data, α, β)

    return Λ_of_t_smooth, wms_of_t_smooth, θ_of_t_smooth
end



function log_cost_to_go_CIR_keep_above_threshold(δ, γ, σ, λ, data, logε::Real; silence = false)
    # println("filter_WF_mem2")

    function prune_keeping_above_logthreshold(Λ_of_t, logwms_of_t)
        Λ_of_t_kept, logwms_of_t_kept = keep_above_threshold(Λ_of_t, logwms_of_t, logε)
        return Λ_of_t_kept, logwms_of_t_kept
    end

    Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t = compute_all_log_cost_to_go_functions_CIR_pruning(δ, γ, σ, λ, data, prune_keeping_above_logthreshold; silence = silence)

    return Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t
end

# function smooth_CIR_keep_above_threshold(δ, γ, σ, λ, data, ε_filter::Real, logε_cost_to_go::Real; silence = false)
#     # println("filter_WF_mem2")
#     α = δ/2
#     β = γ/σ^2
#
#     # function prune_keeping_above_logthreshold(Λ_of_t, logwms_of_t)
#     #     Λ_of_t_kept, logwms_of_t_kept = keep_above_threshold(Λ_of_t, logwms_of_t, log(ε_filter))
#     #     return Λ_of_t_kept, lognormalise(logwms_of_t_kept)
#     # end
#     # This is so because there is no pruning function with log internals at the moment.
#     function prune_keeping_above_threshold(Λ_of_t, wms_of_t)
#         Λ_of_t_kept, wms_of_t_kept = keep_above_threshold(Λ_of_t, wms_of_t, ε_filter)
#         return Λ_of_t_kept, normalise(wms_of_t_kept)
#     end
#
#     Λ_of_t, logwms_of_t, θ_of_t = filter_CIR_pruning_logweights(δ, γ, σ, λ, data, prune_keeping_above_threshold; silence = silence)
#
#     Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t = log_cost_to_go_CIR_keep_above_threshold(δ, γ, σ, λ, data, logε_cost_to_go; silence = silence)
#
#     Λ_of_t_smooth, wms_of_t_smooth, θ_of_t_smooth = merge_filtering_and_cost_to_go_logweights_CIR(Λ_of_t, logwms_of_t, θ_of_t, Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t, data, α, β)
#
#     return Λ_of_t_smooth, wms_of_t_smooth, θ_of_t_smooth
# end

function log_cost_to_go_CIR_keep_fixed_fraction(δ, γ, σ, λ, data, fraction::Float64; silence = false)
    # println("filter_WF_mem2")

    function prune_keeping_fixed_fraction(Λ_of_t, logwms_of_t)
        keep_fixed_fraction_logw(Λ_of_t, logwms_of_t, fraction, logtotal = logsumexp(logwms_of_t))
    end

    Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t = compute_all_log_cost_to_go_functions_CIR_pruning(δ, γ, σ, λ, data, prune_keeping_fixed_fraction; silence = silence)

    return Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t
end

function smooth_CIR_keep_fixed_fraction(δ, γ, σ, λ, data, fraction::Float64; silence = false)
    # println("filter_WF_mem2")
    α = δ/2
    β = γ/σ^2

    # function prune_keeping_above_logthreshold(Λ_of_t, logwms_of_t)
    #     Λ_of_t_kept, logwms_of_t_kept = keep_above_threshold(Λ_of_t, logwms_of_t, log(ε_filter))
    #     return Λ_of_t_kept, lognormalise(logwms_of_t_kept)
    # end
    # This is so because there is no pruning function with log internals at the moment.
    function prune_keeping_fixed_fraction(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_fixed_fraction(Λ_of_t, wms_of_t, fraction)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    Λ_of_t, logwms_of_t, θ_of_t = filter_CIR_pruning_logweights(δ, γ, σ, λ, data, prune_keeping_fixed_fraction; silence = silence)

    Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t = log_cost_to_go_CIR_keep_fixed_fraction(δ, γ, σ, λ, data, fraction::Float64; silence = silence)

    Λ_of_t_smooth, wms_of_t_smooth, θ_of_t_smooth = merge_filtering_and_cost_to_go_logweights_CIR(Λ_of_t, logwms_of_t, θ_of_t, Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t, data, α, β)

    return Λ_of_t_smooth, wms_of_t_smooth, θ_of_t_smooth
end
