using Distributions

function rCIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)
    β = γ/σ^2*exp(2*γ*Dt)/(exp(2*γ*Dt)-1)
    if n == 1
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)))
        return rand(Gamma(ks+δ/2, 1/β))
    else
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)), n)
        return rand.(Gamma.(ks+δ/2, 1/β))
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
