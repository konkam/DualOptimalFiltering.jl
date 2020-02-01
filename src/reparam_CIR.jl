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

function joint_sampler_CIR_reparam_pruning_precompute(data, λ, prior_logpdf, niter, do_the_pruning::Function; final_chain_length = 1000, silence = false, jump_sizes = (0.5, 0.5, 0.5), θ_init = (1.,1.,1.))

    trajectory_chain = Array{Float64,2}(undef, niter, length(data |> keys))
    parameter_chain = Array{Float64,2}(undef, niter, 3)
    times = data |> keys |> collect |> sort

    function unnormalised_logposterior(a, b, σ_prime, trajectory)::Float64
        prior_contribution = prior_logpdf(a, b, σ_prime)
        if isinf(prior_contribution) #Avoids likelihood computation in this case
            return prior_contribution
        else
            return prior_contribution + reparam_joint_loglikelihood_CIR(data, trajectory, times, a, b, σ_prime, λ)::Float64
        end
    end

    θ_it = θ_init

    Jtsym_rand(θ) = rand.(Normal.(0, jump_sizes)) .+ θ

    for it in 1:niter

        print_every = max(1, floor(niter/20))

        if !silence && mod(it, print_every) == 0
            @info "$it iterations out of $niter"
        end

        # Sample trajectory

        #Not using the logfunction because not implemented yet
        θ_it_δγ_param = inverse_reparam_CIR(θ_it...)
        Λ_of_t, wms_of_t, θ_of_t = filter_CIR_pruning(θ_it_δγ_param[1], θ_it_δγ_param[2], θ_it_δγ_param[3], λ, data, do_the_pruning, silence = true)
        Λ_of_t_pruned, wms_of_t_pruned =  prune_all_dicts(Λ_of_t, wms_of_t, do_the_pruning)
        # logwms_of_t = log.(wms_of_t)
        # logwms_pred_of_t = log.(wms_pred_of_t)
        # 1, 1, 1 are dummy arguments
        X_it = sample_1_trajectory_from_joint_smoothing_CIR_precompute(θ_it_δγ_param[1], θ_it_δγ_param[2], θ_it_δγ_param[3], Λ_of_t_pruned, wms_of_t_pruned, θ_of_t, 1, 1, 1, data)

        @debug "iteration $it"
        @debug "trajectory" X_it

        # Metropolis-Hastings step

        θ_it = draw_next_sample(θ_it, Jtsym_rand, st -> unnormalised_logposterior(st[1], st[2], st[3], X_it))

        # Record iteration

        trajectory_chain[it,:] = X_it
        parameter_chain[it,:] = collect(θ_it)

    end

    return parameter_chain, trajectory_chain
end

function joint_sampler_CIR_reparam_keep_fixed_number_precompute(data, λ, prior_logpdf, niter, fixed_number::Int64; final_chain_length = 1000, silence = false, jump_sizes = (0.5, 0.5, 0.5), θ_init = (1.,1.,1.))
    # println("filter_WF_mem2")

    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    joint_sampler_CIR_reparam_pruning_precompute(data, λ, prior_logpdf, niter, prune_keeping_fixed_number; final_chain_length = 1000, silence = silence, jump_sizes = jump_sizes, θ_init = θ_init)

end

function reparam_joint_loglikelihood_CIR(data, trajectory, times, a, b, σ_prime, λ)
    δ, γ, σ = inverse_reparam_CIR(a, b, σ_prime)
    return joint_loglikelihood_CIR(data, trajectory, times, δ, γ, σ, λ)
end
