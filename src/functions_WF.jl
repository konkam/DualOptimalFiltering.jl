using ExactWrightFisher

function prepare_WF_dat_1D_2D(data::Dict{Float64,Array{Int64,2}})
    times = data |> keys |> collect |> sort
    return Dict(zip(times, [vec(data[t]) for t in times])), data
end
function prepare_WF_dat_1D_2D(data::Dict{Float64,Array{Int64,1}})
    times = data |> keys |> collect |> sort
    return data, zip(times, [collect(data[t]') for t in times]) |> Dict
end

function precompute_next_terms_ar!(last_sm_max::Integer, new_sm_max::Integer, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, Δt)
    if last_sm_max == 0
        log_binomial_coeff_ar_offset[1,1] = 0
    end
    for sm in (last_sm_max+1):new_sm_max
        for si in 1:sm
            log_ν_ar[sm, si] = logfirst_term_pmmi_no_alloc(si, sm, sα)
            log_Cmmi_ar[sm, si] = logCmmi_arb(sm, si, Δt, sα)
            log_binomial_coeff_ar_offset[sm+1,si+1] = log_binomial_safe_but_slow(sm, si)
        end
        for si in 0:sm
            log_binomial_coeff_ar_offset[sm+1,si+1] = log_binomial_safe_but_slow(sm, si)
        end
    end
end

function Λ_from_Λ_max(Λ_max) where U <: Integer
    # return Base.Iterators.product((0:Λi_max for Λi_max in Λ_max)...)
    return Base.Iterators.product(map(N -> 0:N, Λ_max)...)
end

indices_of_tree_below(m::Union{AbstractArray{U, 1}, Tuple}) where U <: Integer =  Λ_from_Λ_max(m)


function logfirst_term_pmmi_no_alloc(si::Int64, sm::Int64, sα::Number)
    ##### Equivalent to
    # logλm.((sm-si+1):sm, sα) |> sum
    # but the former version does one allocation
    res = 0
    for s in (sm-si+1):sm
        res += logλm.(s, sα)
    end
    return res
end

function logCmmi_arb(sm::Int64, si::Int64, t::Number, sα::Number)::Float64
    # tmp = [-λm(sm-k, sα)*t - log_denominator_Cmmi_nosign(si, k, sm, sα) for k in 0:si]
    # max_tmp = maximum(tmp)
    return ExactWrightFisher.signed_logsumexp_arb((-λm(sm-k, sα)*t - log_denominator_Cmmi_nosign(si, k, sm, sα) for k in 0:si), (sign_denominator_Cmmi(k) for k in 0:si))[2]
    # return ExactWrightFisher.signed_logsumexp_arb((-λm(sm-k, sα)*t - log_denominator_Cmmi_nosign(si, k, sm, sα) for k in 0:si), take(cycle([1,-1]), si+1))[2]
    # return ExactWrightFisher.signed_logsumexp_arb(-λm.(sm .- (0:si), sα) .* t - log_denominator_Cmmi_nosign.(si, 0:si, sm, sα) , sign_denominator_Cmmi.(0:si) )[2]
end

function logλm(sm::Int64, sα::Number)
    return log(sm) + log(sm + sα - 1) - log(2)
end

function λm(sm::Int64, sα::Number)
    return sm * (sm + sα - 1)/2
end


function log_denominator_Cmmi_nosign(si::Int64, k::Int64, sm::Int64, sα::Number)
    if k==0
        return SpecialFunctions.logfactorial(si) - si*log(2) + log_descending_fact_no0(2*sm + sα - 2, si)
    elseif k==si
        return SpecialFunctions.logfactorial(si) - si*log(2) + log_descending_fact_no0(2*sm + sα - si-1, si)
    else
        return -1.0 .* si * log(2) + SpecialFunctions.logfactorial(k) + SpecialFunctions.logfactorial(si-k) + log_descending_fact_no0(2*sm + sα - 2*k - 2, si-k) + log_descending_fact_no0(2*sm + sα - k - 1, k)
    end
end

function sign_denominator_Cmmi(k::Int64)
    if iseven(k)
        return 1
    else
        return -1
    end
end

function logpmmi_raw_precomputed(i, m, sm::Integer, si::Integer, t::Number, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, log_binomial_coeff_dict::Dict{Tuple{Int64, Int64}, Float64}) where T <: Integer
    return log_ν_dict[(sm, si)] + log_Cmmi_dict[(sm, si)]  + loghypergeom_pdf_using_precomputed(i, m, si, sm, log_binomial_coeff_dict)
end

function logpmmi_raw_precomputed(i, m, sm::Integer, si::Integer, t::Number, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2})
    #Would return an error when called on sm = 0, but this should never occur.
    return log_ν_ar[sm, si] + log_Cmmi_ar[sm, si]  + loghypergeom_pdf_using_precomputed(i, m, si, sm, log_binomial_coeff_ar_offset)
end

function logpmmi_precomputed(i, m, sm::Integer, si::Integer, t::Number, sα::Number, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff) where T <: Integer
    if si==0
        return -λm(sm, sα)*t
    else
        return logpmmi_raw_precomputed(i, m, sm, si, t, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)
    end
end
