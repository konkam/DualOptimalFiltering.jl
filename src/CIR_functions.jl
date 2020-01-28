using Distributions

function rCIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)
    β = γ/σ^2*exp(2*γ*Dt)/(exp(2*γ*Dt)-1)
    if n == 1
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)))
        return rand(Gamma(ks+δ/2, 1/β))
    else
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)), n)
        return rand.(Gamma.(ks .+ δ/2, 1/β))
    end
end

transition_CIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ) = rCIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)

function rec_transition_CIR(Dts, x, δ, γ, σ)
    x_new = transition_CIR(1, Dts[1], x[end], δ, γ, σ)
    if length(Dts) == 1
        return Float64[x; x_new]
    else
        return Float64[x; rec_transition_CIR(Dts[2:end], x_new, δ, γ, σ)]
    end
end

function generate_CIR_trajectory(times, x0, δ, γ, σ)
    Dts = diff(times)
    return rec_transition_CIR(Dts, [x0], δ, γ, σ)
end

function next_wms_from_wms_prime(wms_prime, Λ_prime, y, θ_prime, α)#update
    #Make sure we deal correctly with weights equal to 0
    #Probably harmonise what can be improved in the filtering algorithm
    unnormalised_wms = wms_prime .* DualOptimalFiltering.μπh(Λ_prime, θ_prime, α, y)
    return unnormalised_wms |> DualOptimalFiltering.normalise
end

μπh(m, θ, α, y; λ = 1) = exp.(logμπh(m, θ, α, y; λ = λ))

function logμπh_inside(m, θ, α, y; λ = 1)
    return map(mm -> logμπh(mm::Integer, θ, α, y; λ = λ), m)
end

function logμπh(m::AbstractArray{U, 1}, θ, α, y; λ = 1)  where U<:Integer
    return logμπh_inside(m, θ, α, y; λ = λ)
end

function logμπh(m::Integer, θ, α, y::Integer; λ = 1)
    return y*log(λ) + (α+m)*log(θ) + lgamma_local(m+y+α) - SpecialFunctions.logfactorial(y) - lgamma_local(α+m) - (y+α+m) * log(θ + λ)
end
function logμπh(m::Integer, θ, α, y::AbstractArray{T, 1}; λ = 1) where T <: Integer
    s = sum(y)
    n = length(y)
    return s*log(λ) + (α+m)*log(θ) + lgamma_local(m+s+α) - sum(SpecialFunctions.logfactorial.(y)) - lgamma_local(α+m) - (s+α+m) * log(θ + n*λ)
end

function next_wms_prime_from_wms(wms, Λ, Δt, θ, γ, σ)
    nΛ = length(Λ)
    wms_prime = zeros(maximum(Λ)+1)
    p = γ/σ^2*1/(θ*exp(2*γ*Δt) + γ/σ^2 - θ)
    for k in 1:nΛ
        m = Λ[k]
        for n in 0:m
            idx = n+1
            wms_prime[idx] += wms[k]*pdf(Binomial(m, p), n)
        end
    end
    return wms_prime
end

function d_CIR(m::Integer, n::Integer)
    return m .+ n
end
function e_CIR(θ1::Real, θ2::Real, β::Real)
    return θ1 + θ2 - β
end
