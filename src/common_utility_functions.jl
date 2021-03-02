using SpecialFunctions, IterTools

"Normalises a vector"
function normalise(x::AbstractArray)
    if length(x) == 0
        error("cannot normalise a vector of length 0")
    end
    return x/sum(x)
end

lgamma_local(x) = SpecialFunctions.logabsgamma(x)[1]
const precomputed_lfact = SpecialFunctions.logfactorial.(1:10000)
function lgamma_local(x::Integer)
    if x < 10000 && x > 2
        return precomputed_lfact[x-1]
    else
        # return SpecialFunctions.logfactorial(x)
        return SpecialFunctions.logabsgamma(x)[1]
    end
end

function kmax(x::AbstractArray{T, 1}, k::Integer) where T <: Number
    if length(x) < k
        error("length(x) < $k, cannot take the $k largest elements of a vector of size $(length(x))")
    else
        res = x[(end-k+1):end]
        return kmax_rec(x, k, findmin(res), res)
    end
end

# import Base.length
#
# function length(x::Union{IterTools.Distinct})
#     l = 0
#     for k in x
#         l +=1
#     end
#     return l
# end


function kmax_rec(x::AbstractArray{T, 1}, k::Integer, smallest::Tuple{T,U}, res::AbstractArray{T, 1}) where {T <: Number, U <: Integer}
    # println("x = $x")
    # println("smallest = $smallest")
    # println("res = $res")
    if length(x) == k
        return res
    else
        if x[1] > smallest[1]
            res[smallest[2]] = x[1]
        end
        return kmax_rec(x[2:end], k, findmin(res), res)
    end
end

# function precompute_lgamma_α(α, data)
#     return lgamma_local.(i + α for i in 0:sum(sum.(values(data))))
# end
#
# function precompute_lfactorial(data)
#     return  SpecialFunctions.logfactorial.(1:maximum(maximum.(values(data))))
# end

"Converts a dictionary to a new dictionary with log values"
function convert_weights_to_logweights(weights_dict)
    return Dict(k => log.(v) for (k,v) in weights_dict)
end

"Converts a dictionary to a new dictionary with exponential values"
function convert_logweights_to_weights(weights_dict)
    return Dict(k => exp.(v) for (k,v) in weights_dict)
end

using Memoization
@memoize function log_pochammer_rec(x::Real, n::Integer)
    # @info "n=$n"
    if n==0
        return 0.
    elseif n==1
        return log(x)
    else
        return log(x + n - 1)  + log_pochammer_rec(x, n-1)
    end
end

function test_equal_spacing_of_observations(data; override = false, digits_after_comma_for_time_precision = 4)
    if !override&&(data |> keys |> collect |> sort |> diff |> x -> round.(x; digits = digits_after_comma_for_time_precision) |> unique |> length > 1)
        println(data |> keys |> collect |> sort |> diff |> x -> truncate_float.(x,digits_after_comma_for_time_precision) |> unique)
        error("Think twice about precomputing all terms, as the time intervals are not equal. You can go ahead using the option 'override = true.'")
    end
end

function log_binomial_safe_but_slow(n::Int64, k::Int64)
    @assert n >= 0
    @assert k >= 0
    @assert k <= n
    if k == 0 || k == n
        return 0
    elseif k == 1 || k == n-1
        return log(n)
    else
        return sum(log(i) for i in (n-k+1):n) - sum(log(i) for i in 2:k)
    end
end

function log_descending_fact_no0(x::Real, n::Int64)
    return sum(log(x-i) for i in 0:(n-1))
end

function log_pochammer(x::Real, n::Integer)
    if n==0
        return 0.
    elseif n==1
        return log(x)
    else
        return sum(log(x + i) for i in 0:(n-1))
    end
end

function loghypergeom_pdf_using_precomputed(i, m, si::Integer, sm::Integer, log_binomial_coeff_ar_offset::Array{Float64,2})
    return sum(log_binomial_coeff_ar_offset[m[k]+1,i[k]+1] for k in eachindex(m)) - log_binomial_coeff_ar_offset[sm+1, si+1]
end

function assert_constant_time_step_and_compute_it(data)
    Δts = keys(data) |> collect |> sort |> diff |> unique
    if length(Δts) > 1
        test_equal_spacing_of_observations(data; override = false)
    end
    Δt = mean(Δts)
    return Δt
end

function sample_from_Gamma_mixture(δ, θ, Λ, wms)
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl

    latent_mixture_idx = rand(Categorical(wms))
    return rand(Gamma(δ/2 + Λ[latent_mixture_idx], 1/θ))
end
function create_gamma_mixture_parameters(δ, θ, Λ)
    α = [δ/2 + m for m in Λ]
    β = [θ for m in Λ]
    return α, β
end
function create_gamma_mixture_pdf(δ, θ, Λ, wms)
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    return x -> sum(wms.*Float64[pdf(Gamma(δ/2 + m, 1/θ),x) for m in Λ])
end

function create_dirichlet_mixture(α::Array{T, 1}, Λ::Array{Array{U,1},1}) where {T <: Real, U <:Integer}
    α_mixt = Array{Array{T,1},1}(undef, length(Λ))
    for i in eachindex(Λ)
        α_mixt[i] = α .+ Λ[i]
    end
    return α_mixt
end

create_dirichlet_mixture_parameters = create_dirichlet_mixture #alis for consistency with CIR

function get_quantiles_from_mass(mass)
    qinf = 0.5*(1-mass)
    return (qinf, 1-qinf)
end

function create_Gamma_mixture_density(δ, θ, Λ, wms)
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    return x -> sum(wms.*Float64[pdf(Gamma(δ/2 + m, 1/θ),x) for m in Λ])
end

function compute_quantile_mixture_hpi(δ, θ, Λ, wms, q::Float64)
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    f = x -> sum(wms.*Float64[cdf(Gamma(δ/2 + m, 1/θ),x) for m in Λ])
    return fzero(x -> f(x)-q, 0, 10^9)
end
