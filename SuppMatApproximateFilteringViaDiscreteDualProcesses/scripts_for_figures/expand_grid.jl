function expand_grid(; kws...)
    #code from https://discourse.julialang.org/t/function-like-expand-grid-in-r/4350/20 by user piever (Pietro Vertechi)
    # example: expand_grid(a=1:2, b=1.0:5.0, c=["one", "two", "three", "four"])
    names, vals = keys(kws), values(kws)
    return DataFrame(NamedTuple{names}(t) for t in Iterators.product(vals...))
end
