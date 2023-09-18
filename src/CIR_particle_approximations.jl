using StatsBase, DataStructures 

function CIR_particle_integrated_PD_prediction_step(wms, Λ, Δt, θ, γ, σ; nparts=1000)
    p = γ/σ^2*1/(θ*exp(2*γ*Δt) + γ/σ^2 - θ)
    Λ_wms_prime = Dict{Int64, Float64}()
    for i in 1:nparts
        m = StatsBase.sample(Λ, Weights(wms))
        U = rand(Uniform())
        n = 0# Can replace what comes next by a binomial sample I guess
        w = pdf(Binomial(m, p), n)
        while w < U
            n += 1
            w += pdf(Binomial(m, p), n)
        end
        # ii = 0 
        # n0 = floor((m+1)*p) #max of the binomial
        # ii_threshold = 2*n0-3
        # n = n0
        # w = pdf(Binomial(m, p), n0)
        # while w < U
        #     ii += 1
        #     if ii <= ii_threshold
        #         n = n0 + (-1)^ii*ceil(ii/2)
        #     else
        #         n = n0 + ceil((ii_threshold-1)/2) + i - ii_threshold
        #     end
        #     w += pdf(Binomial(m, p), n)
        # end
        # println(ii)
        if haskey(Λ_wms_prime, n)
            Λ_wms_prime[n] += 1/nparts
        else
            Λ_wms_prime[n] = 1/nparts
        end
    end
    return Λ_wms_prime
end

function birth_nu(i, nmax)
    @assert i<nmax
    @assert i>=0
    nu = zeros(Int64, nmax+1)
    nu[i+1] = -1
    nu[i+2] = 1
    return nu
end

function death_nu(i, nmax)
    @assert i<=nmax
    @assert i>0
    nu = zeros(Int64, nmax+1)
    nu[i+1] = -1
    nu[i] = 1
    return nu
end

function compute_all_rates_BD_CIR(δ, θ, γ, σ, nmax)
    λ = 2*σ^2*(θ-γ/σ^2)
    β = σ^2*δ*(θ-γ/σ^2)
    μ = 2*σ^2*θ

    birth_rates = collect(0:(nmax-1)) .* λ .+ β
    death_rates = collect(1:nmax) .* μ
    birth_nus = birth_nu.(0:(nmax-1), nmax)
    death_nus = death_nu.(1:nmax, nmax)
    nu = Vector{Int64}[birth_nus; death_nus]
    rates = Float64[birth_rates; death_rates]

    nu = hcat(nu...)

    state_indicator = Int64[collect(1:nmax); collect(2:(nmax+1))]

    return nu, rates, state_indicator

end

function F2(x, rates, state_indicator)
    x[state_indicator] .* rates
end

function CIR_particle_BD_prediction_step_gillespie(wms, Λ, Δt, δ, θ, γ, σ, nmax; nparts=1000)
    # The SSA algorithm seems unstable if there are too many events, i.e. if Δt is large and the weights large too.
    particle_sample = rand(Multinomial(nparts, Weights(wms)))
    @assert nmax >= Λ[findall(particle_sample .> 0) |> last]
    state = zeros(Int64, nmax+1)
    for i in 1:length(Λ)
        if particle_sample[i] > 0
            l = Λ[i]
            state[l] = particle_sample[i]
        end
    end
    nu, rates, state_indicator = compute_all_rates_BD_CIR(δ, θ, γ, σ, nmax)

    F_in(x, parms) = F2(x, parms[1:length(rates)], Int64.(parms[(length(rates)+1):end]))

    res = ssa(state, F_in, nu, [rates; state_indicator], Δt)
    res = res.data[end,:]

    # return Iterators.product(map(n -> 0:n, nmax)...), res .* 1/sum(res)
    return Dict(k[1] => k[2] for k in zip(0:nmax, res .* 1/sum(res)))
end


function CIR_particle_PF_approx_prediction_step(wms, Λ, Δt, δ, θ, γ, σ; nparts=1000)
    particles = sample_from_Gamma_mixture(δ, θ, Λ, wms, nparts)
    return rCIR.(1, Δt, particles, δ, γ, σ)
end

function predict_CIR_params_particle_integrated_PD_approx(wms::Array{Ty,1}, δ::Ty, θ::Ty, γ::Ty, σ::Ty, Λ, t::Ty; nparts=1000) where Ty<:Number

    p = γ/σ^2*1/(θ*exp(2*γ*t) + γ/σ^2 - θ)
    θ_new = p * θ * exp(2*γ*t)

    Λ_wms_prime = CIR_particle_integrated_PD_prediction_step(wms, Λ, t, θ, γ, σ; nparts = nparts)

    return θ_new, Λ_wms_prime |> keys |> collect |> sort, getindex.(Ref(Λ_wms_prime), Λ_wms_prime |> keys |> collect |> sort)

end

function predict_CIR_params_particle_BD_approx(wms::Array{Ty,1}, δ::Ty, θ::Ty, γ::Ty, σ::Ty, Λ, t::Ty; nparts=1000) where Ty<:Number

    θ_new = θ # This is specific to the BD process

    Λ_wms_prime = CIR_particle_BD_prediction_step_tavare(wms, Λ, t, δ, θ, γ, σ; nparts = nparts)

    return θ_new, Λ_wms_prime |> keys |> collect |> sort, getindex.(Ref(Λ_wms_prime), Λ_wms_prime |> keys |> collect |> sort)

end

function get_next_filtering_distribution_CIR_particle_approx_PD(current_Λ, current_wms, current_θ, current_time, next_time, δ, γ, σ, λ, next_y; nparts=1000)

    return get_next_filtering_distribution_CIR_particle_approx_PD_BD(current_Λ, current_wms, current_θ, current_time, next_time, δ, γ, σ, λ, next_y, predict_CIR_params_particle_integrated_PD_approx; nparts=nparts)
end

function get_next_filtering_distribution_CIR_particle_approx_BD(current_Λ, current_wms, current_θ, current_time, next_time, δ, γ, σ, λ, next_y; nparts=1000)

    return get_next_filtering_distribution_CIR_particle_approx_PD_BD(current_Λ, current_wms, current_θ, current_time, next_time, δ, γ, σ, λ, next_y, predict_CIR_params_particle_BD_approx; nparts=nparts)
end

function get_next_filtering_distribution_CIR_particle_approx_PD_BD(current_Λ, current_wms, current_θ, current_time, next_time, δ, γ, σ, λ, next_y, predict_function; nparts=1000)
    predicted_θ, predicted_Λ, predicted_wms = predict_function(current_wms, δ, current_θ, γ, σ, current_Λ, next_time-current_time; nparts=nparts)
    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params(predicted_wms, δ, predicted_θ, λ, predicted_Λ, next_y)

    return filtered_θ, filtered_Λ, filtered_wms
end

#kept here for compatibility purposes, remove if no problem seen after a while
get_next_filtering_distribution_particle_approx = get_next_filtering_distribution_CIR_particle_approx_PD


function filter_CIR_particle_integrated_PD_approx(δ, γ, σ, λ, data; silence = false, trim0 = false, nparts=1000)

    filter_CIR_particle_integrated_PD_BD_approx(δ, γ, σ, λ, data, get_next_filtering_distribution_CIR_particle_approx_PD; silence = silence, trim0 = trim0, nparts=nparts)

end

function filter_CIR_particle_integrated_BD_approx(δ, γ, σ, λ, data; silence = false, trim0 = false, nparts=1000)

    filter_CIR_particle_integrated_PD_BD_approx(δ, γ, σ, λ, data, get_next_filtering_distribution_CIR_particle_approx_BD; silence = silence, trim0 = trim0, nparts=nparts)

end


function filter_CIR_particle_integrated_PD_BD_approx(δ, γ, σ, λ, data, next_filtering_distribution_fun; silence = false, trim0 = false, nparts=1000)

    times = keys(data) |> collect |> sort
    Λ_of_t = Dict{Float64, Array{Int64,1}}()
    wms_of_t = Dict{Float64, Array{Float64,1}}()
    θ_of_t = Dict{Float64, Float64}()

    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params([1.], δ, γ/σ^2, λ, [0], data[0])

    Λ_of_t[0] = filtered_Λ
    wms_of_t[0] = filtered_wms # = 1.
    θ_of_t[0] = filtered_θ

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        filtered_θ, filtered_Λ, filtered_wms = next_filtering_distribution_fun(filtered_Λ, filtered_wms, filtered_θ, times[k], times[k+1], δ, γ, σ, λ, data[times[k+1]]; nparts=nparts)
        if trim0
            filtered_Λ, filtered_wms = filtered_Λ[filtered_wms.>0], filtered_wms[filtered_wms.>0]
        end
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
        θ_of_t[times[k+1]] = filtered_θ
    end

    return Λ_of_t, wms_of_t, θ_of_t

end

function g_linear_BD_immigration(t, λ, μ)
    (λ-μ)/(λ-μ*exp((μ-λ)*t))
end

function h_linear_BD_immigration(t, λ, μ)
    (λ-μ)/(λ*exp((λ-μ)*t)-μ)
end

function sim_linear_BD_immigration(i, t, λ, μ, β)
    F = rand(Binomial(i, g_linear_BD_immigration(t, λ, μ)))
    if F==0
        At = 0
    else
        At = rand(NegativeBinomial(F, h_linear_BD_immigration(t, λ, μ))) + F
    end

    n = rand(Poisson(β*t))

    Bt = 0

    for k in 1:n
        tk = rand(Uniform(0, t))
        F = rand(Binomial(1, g_linear_BD_immigration(t-tk, λ, μ)))
        #println(F)
        #println(h_linear_BD_immigration(t-tk, λ, μ))
        if F>0
            Bt += rand(NegativeBinomial(F, h_linear_BD_immigration(t-tk, λ, μ))) + F
        end
    end

    return At + Bt
end

function CIR_particle_BD_prediction_step_tavare(wms, Λ, Δt, δ, θ, γ, σ; nparts=1000)

    λ = 2*σ^2*(θ-γ/σ^2)
    β = σ^2*δ*(θ-γ/σ^2)
    μ = 2*σ^2*θ

    particles = zeros(Int64, nparts)

    for p in eachindex(particles)
        start = sample(Λ, Weights(wms))
        particles[p] = sim_linear_BD_immigration(start, Δt, λ, μ, β)
    end

    res = counter(particles)


    # return Iterators.product(map(n -> 0:n, nmax)...), res .* 1/sum(res)
    return Dict(k => res[k]/nparts for k in keys(res))
end

