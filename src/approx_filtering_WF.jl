function filter_WF_adaptive_precomputation_keep_fixed_number(α, data, fixed_number::Int64; silence = false, trim0 = true)
    # println("filter_WF_mem2")

    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    filter_WF_adaptive_precomputation_ar(α, data, prune_keeping_fixed_number; silence = silence, trim0 = trim0)

end

function filter_WF_adaptive_precomputation_keep_above_threshold(α, data, ε::Float64; silence = false, trim0 = true)
    # println("filter_WF_mem2")

    function prune_keeping_above_threshold(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_above_threshold(Λ_of_t, wms_of_t, ε)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    filter_WF_adaptive_precomputation_ar(α, data, prune_keeping_above_threshold; silence = silence, trim0 = trim0)

end


function filter_WF_adaptive_precomputation_keep_fixed_fraction(α, data, fraction::Float64; silence = false, trim0 = true)
    # println("filter_WF_mem2")

    function prune_keeping_fixed_fraction(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_fixed_fraction(Λ_of_t, wms_of_t, fraction)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end


    filter_WF_adaptive_precomputation_ar(α, data, prune_keeping_fixed_fraction; silence = silence, trim0 = trim0)

end
