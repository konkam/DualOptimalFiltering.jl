using Roots, DataFrames, DataFramesMeta, Query

function compute_marginal_CI(α, Λ_of_t, wms_of_t; mass = 0.95)
    qinfCI, qupCI = get_quantiles_from_mass(mass)
    K = length(α)
    times = Λ_of_t |> keys |> collect |> sort
    Iterators.product(times, 1:K) |>
        collect |>
        x -> map(pr -> DataFrame(time = pr[1],
            margin = pr[2],
            infCI = compute_quantile_beta_mixture(α, Λ_of_t[pr[1]], wms_of_t[pr[1]], pr[2], qinfCI),
            supCI = compute_quantile_beta_mixture(α, Λ_of_t[pr[1]], wms_of_t[pr[1]], pr[2], qupCI)), x) |>
        dflist -> vcat(dflist...)

end

function compute_quantile_beta_mixture(α, Λ, wms, margin, q::Float64)
    function Beta_mixture_cdf(x::Real)
        return sum(wms.*Float64[cdf(Beta(m[margin] + α[margin], sum(m .+ α) - m[margin] - α[margin]),x) for m in Λ])
    end
    return fzero(x -> Beta_mixture_cdf(x)-q, 0, 10^9)
end

function marginal_densities(α, Λ_of_t, wms_of_t)
    K = α |> length
    map(margin -> for_one_marginal(α, margin, Λ_of_t, wms_of_t), 1:K) |>
    dfs -> vcat(dfs...)
end

function for_one_marginal(α, margin, Λ_of_t, wms_of_t)
    times = Λ_of_t |> keys |> collect |> sort

    marginal_psi_t = [create_marginal_beta_mixture(α, Λ_of_t[t], wms_of_t[t], margin) for t in times]
    [DataFrame(x = range(0, stop = 1, length = 100)) |> df -> @transform(df, :dens = f.(:x)) for f in marginal_psi_t] |>
dfs -> map((dfi, ti) -> dfi |> @mutate(time = ti, variable = "x$margin") |> DataFrame, dfs, times) |>
    dfs -> vcat(dfs...)
end

function create_marginal_beta_mixture(α, Λ, wms, margin)
    function Beta_mixture(x::Real)
        return sum(wms.*Float64[pdf(Beta(m[margin] + α[margin], sum(m .+ α) - m[margin] - α[margin]),x) for m in Λ])
    end
    return Beta_mixture
end
