using SpecialFunctions

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
