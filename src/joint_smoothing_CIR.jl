function sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t, data)

    ntimes = length(Λ_of_t)
    times = Λ_of_t |> keys |> collect |> sort
    sample_trajectory = zeros(ntimes)


    xT = sample_from_Gamma_mixture(δ, θ_of_t[times[end]], Λ_of_t[times[end]], exp.(logwms_of_t[times[end]]))
    sample_trajectory[end] = xT
    for i in (ntimes-1):-1:1
        # println("i=$i, xp1=$(sample_trajectory[i+1])")
        ti = times[i]
        tip1 = times[i+1]
        Δt = tip1-ti
        # predictive_dens_ip1 = DualOptimalFiltering.create_Gamma_mixture_density(δ, θ_pred_of_t[tip1], Λ_pred_of_t[tip1], exp.(logwms_pred_of_t[tip1]))(sample_trajectory[i+1])
        predictive_dens_ip1 = compute_normalisation_constant_joint_smoothing_CIR(sample_trajectory[i+1], θ_of_t[ti], θ_primeΔ(Δt, γ, σ), exp.(logwms_of_t[ti]), Λ_of_t[ti], Δt, δ, γ, σ)
        sample_trajectory[i] = backward_sampling_CIR_logw(sample_trajectory[i+1], logwms_of_t[ti], Λ_of_t[ti], predictive_dens_ip1, Δt, δ, γ, σ, θ_of_t[ti])
    end
    return sample_trajectory
end

function sample_1_trajectory_from_joint_smoothing_CIR_precompute(δ, γ, σ, Λ_of_t, wms_of_t, θ_of_t, Λ_pred_of_t, wms_pred_of_t, θ_pred_of_t, data)

    ntimes = length(Λ_of_t)
    times = Λ_of_t |> keys |> collect |> sort
    sample_trajectory = zeros(ntimes)
    mmax = maximum(maximum.(values(Λ_of_t)))

    log_pochammer_precomputed_terms = Array{Float64, 2}(undef, mmax+1, 10^5)#Might need to be updated for larger datasets

    last_k = 0

    xT = sample_from_Gamma_mixture(δ, θ_of_t[times[end]], Λ_of_t[times[end]], wms_of_t[times[end]])
    sample_trajectory[end] = xT
    for i in (ntimes-1):-1:1
        # println("i=$i, xp1=$(sample_trajectory[i+1])")
        ti = times[i]
        tip1 = times[i+1]
        Δt = tip1-ti
        predictive_dens_ip1, last_k = compute_normalisation_constant_adaptive_precomputation(sample_trajectory[i+1], θ_of_t[ti], θ_primeΔ(Δt, γ, σ), wms_of_t[ti], Λ_of_t[ti], Δt, δ, γ, σ, log_pochammer_precomputed_terms, last_k, mmax)
        sample_trajectory[i] = backward_sampling_CIR_precomputed(sample_trajectory[i+1], wms_of_t[ti], Λ_of_t[ti], predictive_dens_ip1, Δt, δ, γ, σ, θ_of_t[ti], log_pochammer_precomputed_terms)
    end
    return sample_trajectory
end


function compute_normalisation_constant_joint_smoothing_CIR(xip1, θ, θ_primeΔt, filtering_weights, filtering_indices, Δt, δ, γ, σ)
    # Careful that this can fail if the increments are not monotonous, i.e. Δs=0 for some values, then starts increasing. Could add a lower limit on k
    s = 0
    Δs = 1
    k=0
    while k <= 10^2 || (Δs > 0 && k <= 10^4)
        xx = pdf(Gamma(δ/2+k, 1/θ_primeΔt*exp(-2*γ*Δt)), xip1)
        Δs = 0
        for idx in eachindex(filtering_indices)
            m = filtering_indices[idx]
            Δs += xx * filtering_weights[idx] * μmθk(k, m, θ, δ, θ_primeΔt)
        end
        s += Δs
        k += 1
        # println("k=$k, Δs=$Δs")
    end
    return s
end

function compute_normalisation_constant_adaptive_precomputation(xip1, θ, θ_primeΔt, filtering_weights, filtering_indices, Δt, δ, γ, σ, precomputed_terms, last_k, mmax)
    # Careful that this can fail if the increments are not monotonous, i.e. Δs=0 for some values, then starts increasing. Could add a lower limit on k
    s = 0
    Δs_rel = 1
    k=0
    while s <= 0 || k <= 10^2 || (Δs_rel > 0 && k <= 10^4)

        if k > last_k
            last_k = k
            precompute_log_pochammer_for_logμmθk!(precomputed_terms, δ, mmax, k; mmin = 0, kmin = last_k)
        end

        xx = pdf(Gamma(δ/2+k, 1/θ_primeΔt*exp(-2*γ*Δt)), xip1)
        Δs = 0
        for idx in eachindex(filtering_indices)
            m = filtering_indices[idx]
            Δs += xx * filtering_weights[idx] * exp(logμmθk5(k, m, θ, δ, θ_primeΔt, precomputed_terms))
        end
        s += Δs
        Δs_rel = Δs/s
        k += 1
    end
    # @info "k=$k, Δs_rel=$Δs_rel"
    return s, last_k
end

function precompute_log_pochammer_for_logμmθk!(precomputed_terms, δ, mmax, kmax; mmin = 0, kmin = 0)
    for m in mmin:mmax
        for k in kmin:kmax
            precomputed_terms[m+1, k+1] = log_pochammer_rec(δ/2+m, k)
        end
    end
end

log_pochammer_precomputed(m, k, precomputed_terms::Array{Float64,2}) = precomputed_terms[m+1,k+1]


function logμmθk5(k, m, θ, δ, θ_primeΔt, precomputed_terms)
    # precomputed
    α = δ/2+m
    β = θ
    λ = θ_primeΔt
    if k==0
        return α*(log(β)-log(λ+β))
    else
        # return sum(log(α + i) for i in 0:(k-1)) - (α+k)*log(λ+β) + k*log(λ) + α*log(β) - lgamma_local(k+1)
        return log_pochammer_precomputed(m, k, precomputed_terms) - (α+k)*log(λ+β) + k*log(λ) + α*log(β) - lgamma_local(k+1)
    end
end

function backward_sampling_CIR_logw(xip1, log_filtering_weights, filtering_indices, νip1_xp1, Δt, δ, γ, σ, θ)
    θ_primeΔt = θ_primeΔ(Δt, γ, σ)
    U = rand()

    # pred_dens_val::Float64 = predictive_dens_ip1(xip1)

    κ, M = select_κM_logw(xip1, θ, θ_primeΔt, U, log_filtering_weights, filtering_indices, Δt, δ, γ, σ, νip1_xp1)

    return rand(Gamma(δ/2+κ+M, 1/(θ+θ_primeΔt)))

end

function backward_sampling_CIR_precomputed(xip1, filtering_weights, filtering_indices, νip1_xp1, Δt, δ, γ, σ, θ, precomputed_terms)
    θ_primeΔt = θ_primeΔ(Δt, γ, σ)
    U = rand()

    # pred_dens_val::Float64 = predictive_dens_ip1(xip1)

    κ, M = select_κM_precomputed(xip1, θ, θ_primeΔt, U, filtering_weights, filtering_indices, Δt, δ, γ, σ, νip1_xp1, precomputed_terms)

    return rand(Gamma(δ/2+κ+M, 1/(θ + θ_primeΔt)))

end

function θ_primeΔ(Δt, γ, σ)
    γ/σ^2 / (exp(2*γ*Δt)-1)
end

function select_κM_logw(xip1, θ, θ_primeΔt, U, log_filtering_weights, filtering_indices, Δt, δ, γ, σ, pred_dens_val)
    k = 0
    logs = -Inf
    while k < 10^4
        logw_tilde_k = logpdf(Gamma(δ/2+k, 1/θ_primeΔt*exp(-2*γ*Δt)), xip1) - log(pred_dens_val)# Careful about the parametrisation of Gamma
        for idx in eachindex(filtering_indices)
            m = filtering_indices[idx]
            logs = logaddexp(logs, logw_tilde_k + log_filtering_weights[idx] + logμmθk(k, m, θ, δ, θ_primeΔt))
            # println("k=$k, m=$m, s=$(exp(logs)), logs=$(logs), U=$U, xip1=$xip1")
            # println("w_tilde_k=$(exp(logw_tilde_k)), wm=$(exp(log_filtering_weights[idx])), mumk=$(μmθk(k, m, θ, δ, θ_primeΔt)), logmumk=$(logμmθk(k, m, θ, δ, θ_primeΔt))")
            # println("mumk=$(μmθk(k, m, θ, δ, θ_primeΔt)), with parameters $((k, m, θ, δ, θ_primeΔt)), max at $(θ/(θ+θ_primeΔt)*(δ/2+m)/(1-θ/(θ+θ_primeΔt)))")
            if exp(logs) >= U
                return k, m
            end
        end
        k += 1
    end
    error("did not manage to select k, weights do not sum to 1")
end

function select_κM_precomputed(xip1, θ, θ_primeΔt, U, filtering_weights, filtering_indices, Δt, δ, γ, σ, pred_dens_val, precomputed_terms)
    k = 0
    s = 0
    while k < 10^4
        w_tilde_k = pdf(Gamma(δ/2+k, 1/θ_primeΔt*exp(-2*γ*Δt)), xip1)/pred_dens_val# Careful about the parametrisation of Gamma
        for idx in eachindex(filtering_indices)
            m = filtering_indices[idx]
            s += w_tilde_k * filtering_weights[idx] * exp(logμmθk5(k, m, θ, δ, θ_primeΔt, precomputed_terms))
            if s >= U
                return k, m
            end
        end
        k += 1
    end
    error("did not manage to select k, weights do not sum to 1")
end

function μmθk(k, m, θ, δ, θ_primeΔt)
    pdf(NegativeBinomial(δ/2 + m, θ/(θ_primeΔt+θ)), k)
end

log_pochammer_mem(a, b) = log_pochammer_rec(a,b)

# function logμmθk4(k, m, θ, δ, θ_primeΔt)
#     # Note that there is some room for optimisation by pre-computing the lgamma(α+k) and using lgamma(α+1) = lgamma(α) + ln(z)
#     α = δ/2+m
#     β = θ
#     λ = θ_primeΔt
#     if k==0
#         return α*(log(β)-log(λ+β))
#     else
#         # return sum(log(α + i) for i in 0:(k-1)) - (α+k)*log(λ+β) + k*log(λ) + α*log(β) - lgamma_local(k+1)
#         return log_pochammer_mem(α, k) - (α+k)*log(λ+β) + k*log(λ) + α*log(β) - lgamma_local(k+1)
#     end
# end

function logμmθk(k, m, θ, δ, θ_primeΔt)
    logpdf(NegativeBinomial(δ/2 + m, θ/(θ_primeΔt+θ)), k)
end
