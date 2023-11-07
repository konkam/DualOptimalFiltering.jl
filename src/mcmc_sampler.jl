using Distributions, CovarianceEstimation, LinearAlgebra

function draw_next_sample(state::T, Jtsym_rand::Function, unnormalised_logposterior::Function) where T
    new_state::T = Jtsym_rand(state)
    logr::Float64 = unnormalised_logposterior(new_state)::Float64 - unnormalised_logposterior(state)::Float64
    if logr > 0
        return new_state
    elseif isinf(logr)
        return state
    else
        u = rand(Uniform())
        if log(u) < logr
            return new_state
        else
            return state
        end
    end
end

function get_mcmc_samples_bare(nsamp, starting_state, Jtsym_rand::Function, unnormalised_logposterior::Function; silence = false, print_acceptance_rate = false)
    chain = Array{typeof(first(starting_state)), 2}(undef, length(starting_state), nsamp+1)
    chain[:,1] = starting_state

    @assert size(Jtsym_rand(starting_state)) == size(starting_state)

    @inbounds for i in 2:(nsamp+1)
        if !silence && mod(i, 10) == 0
            @info "$i iterations out of $nsamp"
        end
        chain[:,i] = draw_next_sample(chain[:,i-1], Jtsym_rand, unnormalised_logposterior)
    end
    if print_acceptance_rate
        println((length(unique(chain[1,:]))-1)/nsamp)
    end
    return chain
end

get_mcmc_samples(nsamp, starting_state, Jtsym_rand_create::Function, unnormalised_logposterior::Function; warmup_percentage = 0.1, final_size = Inf, silence = false, print_acceptance_rate = false) =
get_mcmc_samples_bare(nsamp, starting_state, Jtsym_rand_create(starting_state), unnormalised_logposterior, silence = silence, print_acceptance_rate = print_acceptance_rate) |> c -> discard_warmup(c, warmup_percentage) |> c -> thin_chain(c, final_size)

function discard_warmup(chain, percentage)
    idcut = Int64(round(size(chain, 2) * percentage))
    if idcut == 0
        return chain
    else
        return chain[:, idcut:end]
    end
end

function estimate_step_size(total_number_of_iterations, desired_final_size)
    return floor(total_number_of_iterations/desired_final_size)
end

function thin_chain(chain, final_size)
    step = estimate_step_size(size(chain, 2), final_size)
    if step <= 1.
        return chain
    else
        return chain[:, collect(1:Int64(step):end)]
    end
end

function Jtnorm_create(starting_state)
    function Jtnorm(state)
        return rand(Normal(0,0.5), length(starting_state)) .+ state
    end
    return Jtnorm
end


function estimate_jumping_rule(length_pilot_run, starting_Jtnorm_create, starting_state, unnormalised_logposterior::Function; warmup_percentage = 0.5, print_optimal_jump_size = false, silence = false)

    mcmc_chain = get_mcmc_samples(length_pilot_run, starting_state, starting_Jtnorm_create, unnormalised_logposterior; print_acceptance_rate = false,  warmup_percentage =  warmup_percentage, silence = silence)

    estimated_cov_matrix = cov(AnalyticalNonlinearShrinkage(;corrected=false),mcmc_chain') |> Hermitian |> collect

    optimal_jump_size = estimated_cov_matrix*2.4/sqrt(length(starting_state)) # Gelman et al., Bayesian Data Analysis (2014) p.296

    if print_optimal_jump_size
        println(optimal_jump_size)
    end

    function optimal_kernel(state)
        return rand(MvNormal(optimal_jump_size)) .+ state
    end

    return optimal_kernel
end
