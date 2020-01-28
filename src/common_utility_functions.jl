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

import Base.length

function length(x::Union{IterTools.Distinct})
    l = 0
    for k in x
        l +=1
    end
    return l
end


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

function precompute_lgamma_α(α, data)
    return lgamma_local.(i + α for i in 0:sum(sum.(values(data))))
end

function precompute_lfactorial(data)
    return  SpecialFunctions.logfactorial.(1:maximum(maximum.(values(data))))
end

"Converts a dictionary to a new dictionary with log values"
function convert_weights_to_logweights(weights_dict)
    return Dict(k => log.(v) for (k,v) in weights_dict)
end

"Converts a dictionary to a new dictionary with exponential values"
function convert_logweights_to_weights(weights_dict)
    return Dict(k => exp.(v) for (k,v) in weights_dict)
end
