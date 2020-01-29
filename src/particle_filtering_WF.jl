using FeynmanKacParticleFilters, ExactWrightFisher

function multinomial_logpotential(obs_vector::AbstractArray{T, 1}) where T <: Real
    function pot(state::AbstractArray{T, 1})  where T <: Real
        return logpdf(Multinomial(sum(obs_vector), state), obs_vector)
    end
    return pot
end

function multinomial_logpotential(obs_vector::AbstractArray{T, 2}) where T <: Real
    if (size(obs_vector, 1) == 1) || size(obs_vector, 2) == 1
        return multinomial_logpotential(vec(obs_vector))
    else
        error("must write the multinomial potential function for multiple observations")
    end
end


function create_transition_kernels_WF(data, α_vec::AbstractArray{U, 1}) where {T <: Real, U <: Real}
    sα = sum(α_vec)
    K = length(α_vec)
    function create_Mt(Δt)
        return state -> Wright_Fisher_K_dim_transition_with_t005_approx(state, Δt, α_vec, sα)
    end

    prior = Dirichlet(repeat([1.], K))
    return FeynmanKacParticleFilters.create_transition_kernels(data, create_Mt, prior)
end
