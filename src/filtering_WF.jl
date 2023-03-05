using IterTools, DataStructures

# function update_WF_params(wms::Array{Ty,1}, α::Array{Ty,1}, Λ::Array{Array{Int64,1},1}, y::Array{Int64,2}; debug = false) where Ty<:Number
function update_WF_params(wms::Array{Ty,1}, α::Array{Ty,1}, Λ, y::Array{Int64,2}; debug = false) where Ty<:Number
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
    # function lpga(m)
        sm = sum(m)
        second_term = lgamma_local(sα + sm)
        third_term = sum(lgamma_local.(α + m + nK))
        fourth_term = -lgamma_local(sα + sm + sy)
        fifth_term = -sum(lgamma_local.(α + m))
        return first_term + second_term + third_term + fourth_term + fifth_term
    end

    lpga(m) = lpga(collect(m)) #Not super clean, maybe improve or get rid of tuples altogether

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
        # println("working ? $(length.(unique.(res[1])) == length.(res[1])))" )
    end

    ks = keys(res) |> collect

    return ks, [res[k] for k in ks]

end

function WF_prediction_for_one_m_precomputed(m::Array{Int64,1}, sα::Ty, t::Ty, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = 1) where {Ty<:Number}
    # gm = map(x -> 0:x, m) |> vec |> x -> Iterators.product(x...)
    gm = indices_of_tree_below(m)

    sm = sum(m)


    function fun_n(n)
        i = m.-n
        # println(i)
        si = sum(i)
        return wm*(logpmmi_precomputed(i, m, sm, si, t, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff) |> exp)
    end

    Dict( collect(n) => fun_n(n) for n in gm ) |> Accumulator

end

function filter_WF_adaptive_precomputation_ar(α, data, do_the_pruning::Function; silence = false, return_precomputed_terms = false, trim0 = false)
    # println("filter_WF_mem2")

    # @assert length(α) == length(data[collect(keys(data))[1]])
    # println("$α, $(length(data[data |> keys |> first]))")
    @assert length(α) == length(data[data |> keys |> first])
    Δt = assert_constant_time_step_and_compute_it(data)

    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)

    sα = sum(α)
    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()

    filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data[times[1]])
    Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)
    
    if trim0
        Λ_pruned, wms_pruned = Λ_pruned[wms_pruned.>0], wms_pruned[wms_pruned.>0]
    end

    Λ_of_t[times[1]] = filtered_Λ
    wms_of_t[times[1]] = filtered_wms
    new_sm_max = maximum(sum.(Λ_pruned))
    precompute_next_terms_ar!(0, new_sm_max, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sα, Δt)
    sm_max_so_far = new_sm_max

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        last_sm_max = maximum(sum.(Λ_pruned))
        new_sm_max = last_sm_max + sum(data[times[k+1]])

        if sm_max_so_far < new_sm_max
            precompute_next_terms_ar!(sm_max_so_far, new_sm_max, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sα, Δt)
            sm_max_so_far = max(sm_max_so_far,new_sm_max)
        end

        filtered_Λ, filtered_wms = get_next_filtering_distribution_precomputed(Λ_pruned, wms_pruned, times[k], times[k+1], α, sα, data[times[k+1]], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)
        Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)
        if trim0
            Λ_pruned, wms_pruned = Λ_pruned[wms_pruned.>0], wms_pruned[wms_pruned.>0]
        end
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
    end

    if return_precomputed_terms
        return Λ_of_t, wms_of_t, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far
    else
        return Λ_of_t, wms_of_t
    end

end

function get_next_filtering_distribution_precomputed(current_Λ, current_wms, current_time, next_time, α, sα, next_y, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)

    predicted_Λ, predicted_wms = predict_WF_params_precomputed(current_wms, sα, current_Λ, next_time-current_time, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)
    filtered_Λ, filtered_wms = update_WF_params(predicted_wms, α, predicted_Λ, next_y)

    return filtered_Λ, filtered_wms
end


function filter_WF(α, data; silence = false, return_precomputed_terms = false, trim0 = true)

    return filter_WF_adaptive_precomputation_ar(α, data, (x, y) -> (x,y); silence = silence, return_precomputed_terms = return_precomputed_terms, trim0 = trim0)
end
