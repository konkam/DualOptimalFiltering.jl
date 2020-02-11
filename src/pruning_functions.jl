function keep_above_threshold(Λ_of_t, wms_of_t, ε)
    # Might be a good idea to write a specialised function for generators
    # Anyway this has two passes on the weight vector, could be done in one pass
    wms_of_t_c = collect(wms_of_t)
    Λ_of_t_c = collect(Λ_of_t)
    Λ_of_t_kept = [Λ_of_t_c[i] for i in eachindex(Λ_of_t_c) if wms_of_t_c[i] >= ε]
    wms_of_t_kept = [wms_of_t_c[i] for i in eachindex(Λ_of_t_c) if wms_of_t_c[i] >= ε]
    return Λ_of_t_kept, wms_of_t_kept
end

function keep_fixed_number_of_weights(Λ_of_t, wms_of_t, k)
    if length(wms_of_t) <= k
        return Λ_of_t, wms_of_t
    else
        #This is intended to work also with iterators
        #Some more work might be needed, but it will only be implemented if this part of the code is critical for performance.
        wms_of_t_c = collect(wms_of_t)
        last_w = kmax(wms_of_t_c, k) |> minimum #smallest weight kept
        return keep_above_threshold(collect(Λ_of_t), wms_of_t_c, last_w)
    end
end

function keep_fixed_fraction(Λ_of_t, wms_of_t, fraction; total = 1.)
    #This is not guaranteed to keep exactly the correct fraction, should be called "keep at least fraction". This keeps more when there are several weights equal to the threshold
    sorted_wms_of_t = sort(wms_of_t, rev = true)
    cumulative_sum::Float64 = 0.
    i::Int64 = 0
    max_i = length(sorted_wms_of_t)
    # @show max_i
    frac_times_total::Float64 = fraction * total
    @inbounds while i <= max_i && cumulative_sum < frac_times_total
        i +=1
        # @show i
        cumulative_sum += sorted_wms_of_t[i]
    end
    threshold = sorted_wms_of_t[i]
    return keep_above_threshold(Λ_of_t, wms_of_t, threshold)
end

function keep_fixed_fraction_logw(Λ_of_t, logwms_of_t, fraction; logtotal = 0.)
    #This is not guaranteed to keep exactly the correct fraction, should be called "keep at least fraction". This keeps more when there are several weights equal to the threshold
    sorted_logwms_of_t = sort(logwms_of_t, rev = true)
    cumulative_logsum::Float64 = -Inf
    i::Int64 = 0
    max_i = length(sorted_logwms_of_t)
    # @show max_i
    logfrac_times_total::Float64 = log(fraction) + logtotal
    @inbounds while i <= max_i && cumulative_logsum < logfrac_times_total
        i +=1
        # @show i
        cumulative_logsum = logaddexp(cumulative_logsum, sorted_logwms_of_t[i])
    end
    logthreshold = sorted_logwms_of_t[i]
    return keep_above_threshold(Λ_of_t, logwms_of_t, logthreshold)
end

function prune_keeping_fixed_fraction_log_wms(Λ_of_t, log_wms_of_t, fraction::Real)
    Λ_of_t_kept, wms_of_t_kept = keep_fixed_fraction(Λ_of_t, exp.(log_wms_of_t), fraction)
    return Λ_of_t_kept, log.(normalise(wms_of_t_kept))
end

function prune_keeping_fixed_number_log_wms(Λ_of_t, log_wms_of_t, fixed_number::Integer)
    Λ_of_t_kept, log_wms_of_t_kept = keep_fixed_number_of_weights(Λ_of_t, log_wms_of_t, fixed_number)
    return Λ_of_t_kept, log_wms_of_t_kept .- logsumexp(log_wms_of_t_kept)
end

function prune_keeping_above_threshold_log_wms(Λ_of_t, log_wms_of_t, ε::Real)
    Λ_of_t_kept, log_wms_of_t_kept = keep_above_threshold(Λ_of_t, log_wms_of_t, log(ε))
    return Λ_of_t_kept, log_wms_of_t_kept .- logsumexp(log_wms_of_t_kept)
end

function prune_all_dicts(Λ_of_ts, wms_of_ts, pruning_function)
    Λ_of_ts_pruned = Dict()
    wms_of_ts_pruned = Dict()
    for k in keys(Λ_of_ts)
        Λ_of_ts_pruned[k], wms_of_ts_pruned[k] = pruning_function(Λ_of_ts[k], wms_of_ts[k])
    end
    return Λ_of_ts_pruned, wms_of_ts_pruned
end
function prune_all_dicts(Λ_of_ts::Dict{Float64, Array{Int64,1}}, wms_of_ts::Dict{Float64, Array{Float64,1}}, pruning_function::Function)
    Λ_of_ts_pruned = Dict{Float64, Array{Int64,1}}()
    wms_of_ts_pruned = Dict{Float64, Array{Float64,1}}()
    for k in keys(Λ_of_ts)
        Λ_of_ts_pruned[k], wms_of_ts_pruned[k] = pruning_function(Λ_of_ts[k], wms_of_ts[k])
    end
    return Λ_of_ts_pruned, wms_of_ts_pruned
end
