function WF_loglikelihood_from_adaptive_filtering(α, data, do_the_pruning::Function; silence = false)

    @assert length(α) == length(data[collect(keys(data))[1]])
    Δts = keys(data) |> collect |> sort |> diff |> unique
    if length(Δts) > 1
        test_equal_spacing_of_observations(data; override = false)
    end
    Δt = mean(Δts)


    sα = sum(α)
    times = keys(data) |> collect |> sort
    data, data_2D_array =  prepare_WF_dat_1D_2D(data)

    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)
    μν_prime_im1 = Array{Float64,1}(undef, length(times))

    Λ_of_t = Dict()
    wms_of_t = Dict()

    μν_prime_im1[1] = logμπh_WF(α, zeros(Int64, length(α)), data[times[1]])

    filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data_2D_array[times[1]])
    Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)

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

        filtered_Λ, filtered_wms, μν_prime_im1[k+1] = get_next_WF_filtering_distribution_and_loglikelihood_precomputed(Λ_pruned, wms_pruned, times[k], times[k+1], α, sα, data_2D_array[times[k+1]], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)
        Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
    end

    # return Λ_of_t, wms_of_t
    log_lik_terms = Dict(zip(times, cumsum(μν_prime_im1)))
    return log_lik_terms

end

function logμπh_WF(α::AbstractArray{T, 1}, m::Union{AbstractArray{U, 1}, Tuple}, y::AbstractArray{U, 1}) where {T <: Real, U <: Integer}
    ## Needs to be written for multiple observations too
    sy = sum(y)
    sα = sum(α)
    return SpecialFunctions.logfactorial(sy) - sum(SpecialFunctions.logfactorial.(y)) + sum(log_pochammer.(α .+ m, y)) - log_pochammer(sα + sum(m), sy)
end
