using StatsFuns

function log_likelihood_CIR(δ, γ, σ, λ, data)
    times = data |> keys |> collect |> sort

    α = δ/2#Alternative parametrisation

    res = Dict()

    res[times[1]] = logμπh_param_δγ(0, γ/σ^2, δ, data[times[1]])

    #Prior
    Λ_prime = [0]
    θ_prime = γ/σ^2

    #1st update
    Λ = next_Λ_from_Λ_prime(Λ_prime, data[times[1]], t_CIR)
    θ = θ_from_θ_prime(data[times[1]], θ_prime, T_CIR)
    log_wms = [0]

    #1st prediction
    Λ_prime = Λ_prime_1D(Λ)
    log_wms_prime = next_log_wms_prime_from_log_wms1D(log_wms, Λ, times[2]-times[1], θ, γ, σ)
    θ_prime = θ_prime_from_θ_CIR(θ, times[2]-times[1], γ, σ)

    res[times[2]] = res[times[1]] + logμν_i_minus_1(Λ_prime, log_wms_prime, θ_prime, data[times[2]], δ)

    for i in 2:(length(times)-1)
        t = times[i]
        next_t = times[i+1]
        Δt = next_t-t

        #Update
        Λ = next_Λ_from_Λ_prime(Λ_prime, data[t], t_CIR)
        θ = θ_from_θ_prime(data[t], θ_prime, T_CIR)
        log_wms = next_logwms_from_log_wms_prime1D(log_wms_prime, Λ_prime, data[t], θ_prime, α)

         #Prediction
        Λ_prime = Λ_prime_1D(Λ)
        θ_prime = θ_prime_from_θ_CIR(θ, Δt, γ, σ)
        log_wms_prime = next_log_wms_prime_from_log_wms1D(log_wms, Λ, Δt, θ, γ, σ)

        res[next_t] = res[t] + logμν_i_minus_1(Λ_prime, log_wms_prime, θ_prime, data[next_t], δ)

    end

    return res
end


logμπh_param_δγ(m, θ, δ, y; λ = 1) = logμπh(m, θ, δ/2, y; λ = λ)


function log_likelihood_CIR_keep_fixed_number(δ, γ, σ, λ, data, fixed_number::Int64; silence = false)
    # println("filter_WF_mem2")

    prune_keeping_fixed_number_log_wms_(Λ_of_t, log_wms_of_t) =
        prune_keeping_fixed_number_log_wms(Λ_of_t, log_wms_of_t, fixed_number)

    log_likelihood_pruning(δ, γ, σ, λ, data, prune_keeping_fixed_number_log_wms_; silence = silence)

end

function log_likelihood_CIR_keep_above_threshold(δ, γ, σ, λ, data, ε::Float64; silence = false)
    # println("filter_WF_mem2")

    prune_keeping_above_threshold_log_wms_(Λ_of_t, log_wms_of_t) = prune_keeping_above_threshold_log_wms(Λ_of_t, log_wms_of_t, ε)

    log_likelihood_pruning(δ, γ, σ, λ, data, prune_keeping_above_threshold_log_wms_; silence = silence)

end

function log_likelihood_CIR_fixed_fraction(δ, γ, σ, λ, data, fraction::Float64; silence = false)
    # println("filter_WF_mem2")
    prune_keeping_fixed_fraction_log_wms_(Λ_of_t, log_wms_of_t) = prune_keeping_fixed_fraction_log_wms(Λ_of_t, log_wms_of_t, fraction)

    log_likelihood_pruning(δ, γ, σ, λ, data, prune_keeping_fixed_fraction_log_wms_; silence = silence)

end

function log_likelihood_pruning(δ, γ, σ, λ, data, do_the_pruning_log_wms::Function; silence = false)
    times = data |> keys |> collect |> sort

    α = δ/2#Alternative parametrisation

    res = Dict()

    res[times[1]] = logμπh_param_δγ(0, γ/σ^2, δ, data[times[1]])

    #Prior
    Λ_prime = [0]
    θ_prime = γ/σ^2

    #1st update
    Λ = next_Λ_from_Λ_prime(Λ_prime, data[times[1]], t_CIR)
    θ = θ_from_θ_prime(data[times[1]], θ_prime, T_CIR)
    log_wms = [0]

    pruned_Λ, pruned_log_wms = do_the_pruning_log_wms(Λ, log_wms)

    #1st prediction
    Λ_prime = Λ_prime_1D(pruned_Λ)
    log_wms_prime = next_log_wms_prime_from_log_wms1D(pruned_log_wms, pruned_Λ, times[2]-times[1], θ, γ, σ)
    θ_prime = θ_prime_from_θ_CIR(θ, times[2]-times[1], γ, σ)

    res[times[2]] = res[times[1]] + logμν_i_minus_1(Λ_prime, log_wms_prime, θ_prime, data[times[2]], δ)

    for i in 2:(length(times)-1)
        t = times[i]
        next_t = times[i+1]
        Δt = next_t-t

        #Update
        Λ = next_Λ_from_Λ_prime(Λ_prime, data[t], t_CIR)
        θ = θ_from_θ_prime(data[t], θ_prime, T_CIR)
        log_wms = next_logwms_from_log_wms_prime1D(log_wms_prime, Λ_prime, data[t], θ_prime, α)

        #Pruning
        pruned_Λ, pruned_log_wms = do_the_pruning_log_wms(Λ, log_wms)


         #Prediction
        Λ_prime = Λ_prime_1D(pruned_Λ)
        θ_prime = θ_prime_from_θ_CIR(θ, Δt, γ, σ)
        log_wms_prime = next_log_wms_prime_from_log_wms1D(pruned_log_wms, pruned_Λ, Δt, θ, γ, σ)

        res[next_t] = res[t] + logμν_i_minus_1(Λ_prime, log_wms_prime, θ_prime, data[next_t], δ)

    end

    return res
end

function t_CIR(y::Number, m::Integer)
    return m + y
end
function t_CIR(y::Number, m::Array{U, 1}) where U<:Integer
    return m .+ y
end
function t_CIR(y::Array{T, 1}, m::Integer) where T<:Number
    return m + sum(y)
end
function t_CIR_inside(y::Array{T, 1}, m) where T<:Number
    return m .+ sum(y)
end
function t_CIR(y::Array{T, 1}, m::Array{U, 1}) where {T<:Number, U<:Integer}
    return t_CIR_inside(y::Array{T, 1}, m)
end
function t_CIR(y::Array{T, 1}, m::UnitRange{Int64}) where T<:Number
    return t_CIR_inside(y::Array{T, 1}, m)
end

function Λ_prime_1D(Λ::Union{Array{T,1}, UnitRange{Int64}}) where T<:Integer
    return 0:maximum(Λ)
end

function next_Λ_from_Λ_prime(Λ_prime, y, t)
    return t(y, Λ_prime)
end

function θ_prime_from_θ_CIR(θ, Δt, γ, σ)
    return γ/σ^2 * θ*exp(2*γ*Δt)/(θ*exp(2*γ*Δt)+γ/σ^2-θ)
end

function θ_from_θ_prime(y, θ_prime, T)
    return T(y, θ_prime)
end

T_CIR(y, θ; λ = 1) = θ + λ*length(y)


function next_logwms_from_log_wms_prime1D(log_wms_prime, Λ_prime, y, θ_prime, α)#update
    #Make sure we deal correctly with weights equal to 0
    #Probably harmonise what can be improved in the filtering algorithm
    unnormalised_log_wms = log_wms_prime .+ logμπh(Λ_prime, θ_prime, α, y)
    return unnormalised_log_wms .- StatsFuns.logsumexp(unnormalised_log_wms)
end

function next_log_wms_prime_from_log_wms1D(log_wms, Λ, Δt, θ, γ, σ)
    #The speed gain seems like a modest 20%, but much less allocations though.
    nΛ = length(Λ)
    maxΛ = maximum(Λ)
    wms_prime = zeros(maxΛ+1)
    all_u = zeros(maxΛ+1)
    all_u .= -Inf
    p = γ/σ^2*1/(θ*exp(2*γ*Δt) + γ/σ^2 - θ)
    @inbounds for k in eachindex(Λ)
        m = Λ[k]
        for n in 0:m
            idx = n+1

            x = log_wms[k] + logpdf(Binomial(m, p), n)
            if x <= all_u[idx]
                wms_prime[idx] += exp(x - all_u[idx])
            else
                wms_prime[idx] *= exp(all_u[idx] - x)
                wms_prime[idx] += 1.0
                all_u[idx] = x
            end
        end
    end
    # return @. log(wms_prime) + all_u
    return log.(wms_prime) .+ all_u
end

function logμν_i_minus_1(Λ_prime_i_minus_1, log_wms_prime_i_minus_1, θ_prime_i_minus_1, yi, δ)
    StatsFuns.logsumexp(log_wms_prime_i_minus_1 .+ logμπh_param_δγ(Λ_prime_i_minus_1, θ_prime_i_minus_1, δ, yi))
end
