function reparam_CIR(δ, γ, σ)
    a = 2*γ
    b = δ*σ^2/(2*γ)
    σ_prime = 2*σ
    return a, b, σ_prime
end

function inverse_reparam_CIR(a, b, σ_prime)
    γ = a/2
    δ = 4*a*b/σ_prime^2
    σ = σ_prime/2
    return δ, γ, σ
end

function joint_sampler_CIR_reparam_pruning_precompute(data, λ, prior_logpdf, niter, do_the_pruning::Function; final_chain_length = 1000, silence = false, jump_size = 0.5*Matrix(I,3,3), θ_init = [1.,1.,1.])

    joint_sampler_CIR_pruning_precompute(data, λ, prior_logpdf, θ_init, niter, do_the_pruning::Function; final_chain_length = final_chain_length, silence = silence, jump_size = jump_size, reparam = true)
end

function joint_sampler_CIR_reparam_keep_fixed_number_precompute(data, λ, prior_logpdf, niter, fixed_number::Int64; final_chain_length = 1000, silence = false, jump_size = 0.5*Matrix(I,3,3), θ_init = [1.,1.,1.])
    # println("filter_WF_mem2")

    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    joint_sampler_CIR_reparam_pruning_precompute(data, λ, prior_logpdf, niter, prune_keeping_fixed_number; final_chain_length = 1000, silence = silence, jump_size = jump_size, θ_init = θ_init)

end

function reparam_joint_loglikelihood_CIR(data, trajectory, times, a, b, σ_prime, λ)
    δ, γ, σ = inverse_reparam_CIR(a, b, σ_prime)
    return joint_loglikelihood_CIR(data, trajectory, times, δ, γ, σ, λ)
end
