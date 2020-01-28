using IterTools, DataStructures

function update_WF_params(wms::Array{Ty,1}, α::Array{Ty,1}, Λ::Array{Array{Int64,1},1}, y::Array{Int64,2}; debug = true) where Ty<:Number
    #y is a matrix of dimension J*K, with K the dimension of the process
    # and J the number of observations
    # Julia is in row major, so the first index indicates the row (index of observation)
    # and the second the column (index of the dimension) (as in matrix mathematical notation)
    @assert length(wms) == size(Λ, 1)

    nJ = sum(y, dims = 2) |> vec#sum_j=1^K n_ij
    nK = sum(y, dims = 1) |> vec#sum_i=1^J n_ij
    sy = sum(y)
    J = size(y,1)
    sα = sum(α)


    first_term = sum(SpecialFunctions.logfactorial.(nJ) - sum(SpecialFunctions.logfactorial.(y), dims = 2))


    function lpga(m::Array{Int64,1})
        sm = sum(m)
        second_term = lgamma_local(sα + sm)
        third_term = sum(lgamma_local.(α + m + nK))
        fourth_term = -lgamma_local(sα + sm + sy)
        fifth_term = -sum(lgamma_local.(α + m))
        return first_term + second_term + third_term + fourth_term + fifth_term
    end

    filter_ = wms .== 0

    lwms_hat = log.(wms) .+ map(lpga, Λ)

    wms_hat = exp.(lwms_hat)
    wms_hat[filter_] .= 0
    wms_hat = wms_hat |> normalise

    if debug&&any(isnan, wms_hat)
        println("NAs in update step")
        println("wms")
        println(wms)
        println("pga")
        println(map(pga, Λ))
        println("wms_hat")
        println(wms_hat)
    end

    return [m .+ nK for m in Λ], wms_hat
end

function get_next_WF_filtering_distribution_and_loglikelihood_precomputed(current_Λ, current_wms, current_time, next_time, α, sα, next_y, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)
    predicted_Λ, predicted_wms = predict_WF_params_precomputed(current_wms, sα, current_Λ, next_time-current_time, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)

    μν_prime_im1 = logsumexp(log(predicted_wms[k]) + logμπh_WF(α, predicted_Λ[k], vec(next_y)) for k in eachindex(predicted_Λ))# for the time being, because one only deal with single observations

    filtered_Λ, filtered_wms = update_WF_params(predicted_wms, α, predicted_Λ, next_y)

    return filtered_Λ, filtered_wms, μν_prime_im1
end

function predict_WF_params_precomputed(wms::Array{Ty,1}, sα::Ty, Λ::Array{Array{Int64,1},1}, t::Ty, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = 1) where {Ty<:Number}

    res = Accumulator{Array{Int64,1}, Float64}()

    for k in eachindex(Λ)
        res = merge(res, WF_prediction_for_one_m_precomputed(Λ[k], sα, t, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = wms[k]))
    end

    ks = keys(res) |> collect

    return ks, [res[k] for k in ks]

end

function WF_prediction_for_one_m_precomputed(m::Array{Int64,1}, sα::Ty, t::Ty, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = 1) where {Ty<:Number}
    # gm = map(x -> 0:x, m) |> vec |> x -> Iterators.product(x...)
    gm = indices_of_tree_below(m)

    function fun_n(n)
        i = m.-n
        # println(i)
        si = sum(i)
        sm = sum(m)
        return wm*(logpmmi_precomputed(i, m, sm, si, t, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff) |> exp)
    end

    Dict( collect(n) => fun_n(n) for n in gm ) |> Accumulator

end
