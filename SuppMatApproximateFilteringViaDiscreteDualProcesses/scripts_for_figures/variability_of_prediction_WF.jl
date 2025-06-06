include("expand_grid.jl")

function add_local(u, v)
    return u .+ v
end

function compute_moment_errors_Dirichlet_mixtures(wms_1, α_1, wms_2, α_2)
    logdist = NaN 
    marginal_first_moments_1 = reduce(add_local, (wms_1 .* (α_i ./ sum(α_i) for α_i in α_1)))
    marginal_first_moments_2 = reduce(add_local, (wms_2 .* (α_i ./ sum(α_i) for α_i in α_2)))
    marginal_second_moments_1 = reduce(add_local, (wms_1 .* (α_i .* (α_i .+ 1) ./ (sum(α_i)*(sum(α_i)+1)) for α_i in α_1)))
    marginal_second_moments_2 = reduce(add_local, (wms_2 .* (α_i .* (α_i .+ 1) ./ (sum(α_i)*(sum(α_i)+1)) for α_i in α_2)))

    marginal_first_moments_mean_abs_error = abs.(marginal_first_moments_1 - marginal_first_moments_2) |> mean
    marginal_second_moments_mean_abs_error = abs.(marginal_second_moments_1 - marginal_second_moments_2) |> mean
    marginal_sds_mean_abs_error = abs.(sqrt.(marginal_second_moments_1 .- marginal_first_moments_1 .^ 2) - sqrt.(marginal_second_moments_2 .- marginal_first_moments_2 .^ 2)) |> mean

    return logdist, marginal_first_moments_mean_abs_error, marginal_second_moments_mean_abs_error, marginal_sds_mean_abs_error
end

function compute_moment_errors_WF_particle_approx(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=100, WF_particle_approxfun = DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step)

    Λ_2, wms_2 = WF_particle_approxfun(wms_start, α, Λ_start, forecast_time; nparts=nparts)

    α_2 = DualOptimalFiltering.create_dirichlet_mixture(α, Λ_2)
    α_ref = DualOptimalFiltering.create_dirichlet_mixture(α, Λ_ref)

    return compute_moment_errors_Dirichlet_mixtures(wms_ref, α_ref, wms_2, α_2)

end

function auxiliaryfun_PD_Gillespie(wms_start, α, Λ_start, forecast_time; nparts=100)
    sα = sum(α)
    DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step(wms_start, sα, Λ_start, forecast_time; nparts = nparts)
end

function compute_moment_errors_WF_particle_PD_Gillespie(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=100)

    println("PD Gillespie")

    return compute_moment_errors_WF_particle_approx(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=nparts, WF_particle_approxfun = auxiliaryfun_PD_Gillespie)
end

function create_auxiliaryfun_PD_integrated(α, Λ_start, forecast_time)

    sα = sum(α)

    smmax = Λ_start |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)
    DualOptimalFiltering.precompute_next_terms_ar!(0, smmax, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, forecast_time)

    function auxiliaryfun_PD_integrated(wms_start, α, Λ_start, forecast_time; nparts=100)
        return DualOptimalFiltering.WF_particle_integrated_PD_prediction_step(wms_start, sα, Λ_start, forecast_time, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = nparts)
    end

    return auxiliaryfun_PD_integrated
end


function compute_moment_errors_WF_particle_PD_integrated(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=100)

    auxiliaryfun_PD_integrated = create_auxiliaryfun_PD_integrated(α, Λ_start, forecast_time)

    println("PD integrated")

    return compute_moment_errors_WF_particle_approx(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=nparts, WF_particle_approxfun = auxiliaryfun_PD_integrated)

end

function auxiliaryfun_BD_Moran_Gillespie(wms_start, α, Λ_start, forecast_time; nparts=nparts)
    DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_Moran_gillespie(wms_start, Λ_start, forecast_time, α; nparts = nparts)
end

function compute_moment_errors_WF_particle_BD_Moran_Gillespie(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=100)

    println("BD_Moran_Gillespie")

     return compute_moment_errors_WF_particle_approx(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=nparts, WF_particle_approxfun = auxiliaryfun_BD_Moran_Gillespie)
end

function auxiliaryfun_BD_WF_gillespie(wms_start, α, Λ_start, forecast_time; nparts=nparts)
    DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_WF_gillespie(wms_start, Λ_start, forecast_time, α; nparts = nparts)
end

function compute_moment_errors_WF_particle_BD_WF_Gillespie(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=100)

    println("BD_WF_Gillespie")


     return compute_moment_errors_WF_particle_approx(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=nparts, WF_particle_approxfun = auxiliaryfun_BD_WF_gillespie)
end

function auxiliaryfun_BD_WF_diffusion(wms_start, α, Λ_start, forecast_time; nparts=nparts)
    DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_WF_diffusion(wms_start, Λ_start, forecast_time, α; nparts = nparts)
end

function compute_moment_errors_WF_particle_BD_WF_diffusion(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=100)

    println("BD_WF_diffusion")


     return compute_moment_errors_WF_particle_approx(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=nparts, WF_particle_approxfun = auxiliaryfun_BD_WF_diffusion)
end

function compute_moment_errors_WF_PF(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=100)

    println("WF_PF")

    particle_PF_approx = DualOptimalFiltering.WF_particle_boostrap_prediction_step(wms_start, Λ_start, forecast_time, α; nparts=nparts)


    α_ref = DualOptimalFiltering.create_dirichlet_mixture(α, Λ_ref)

    marginal_first_moments_ref = reduce(add_local, (wms_ref .* (α_i ./ sum(α_i) for α_i in α_ref)))
    marginal_second_moments_ref = reduce(add_local, (wms_ref .* (α_i .* (α_i .+ 1) ./ (sum(α_i)*(sum(α_i)+1)) for α_i in α_ref)))

    marginal_sds_ref = marginal_second_moments_ref .- marginal_first_moments_ref .^ 2

    marginal_first_moments_PF = mean(particle_PF_approx, dims = 2)
    marginal_second_moments_PF = var(particle_PF_approx, dims = 2) .+ marginal_first_moments_PF .^ 2
    marginal_sds_PF = std(particle_PF_approx, dims = 2)

    marginal_first_moments_mean_abs_error = abs.(marginal_first_moments_ref - marginal_first_moments_PF) |> mean
    marginal_second_moments_mean_abs_error = abs.(marginal_second_moments_ref - marginal_second_moments_PF) |> mean
    marginal_sds_mean_abs_error = abs.(marginal_sds_ref .- marginal_sds_PF) |> mean

    logdist = NaN

    return logdist, marginal_first_moments_mean_abs_error, marginal_second_moments_mean_abs_error, marginal_sds_mean_abs_error
end

function nth(v, n)
    return v[n]
end

second(v) = nth(v, 2)
third(v) = nth(v, 3)
fourth(v) = nth(v, 4)


function average_errors_one_time_horizon(forecast_time, nrep, wms_start, Λ_start, α, nparts, errors_fun)

    sα = sum(α)

    smmax = Λ_start |> sum |> sum

    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)
    DualOptimalFiltering.precompute_next_terms_ar!(0, smmax, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, forecast_time)

    Λ_ref, wms_ref = DualOptimalFiltering.predict_WF_params_precomputed(wms_start, sα, Λ_start, forecast_time, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    res = map(i -> errors_fun(wms_ref, Λ_ref, wms_start, Λ_start, α, forecast_time; nparts=nparts), 1:nrep)

    DataFrame(expected_log_L2_error = logsumexp(first.(res)) - log(nrep), log_L2_CI_size = 4*std(first.(res))/sqrt(nrep),
    expected_first_moment_error = mean(second.(res)), first_moment_CI_size = 4*std(second.(res))/sqrt(nrep),
    expected_second_moment_error = mean(third.(res)), second_moment_CI_size = 4*std(third.(res))/sqrt(nrep), 
    expected_sd_error = mean(fourth.(res)), sd_CI_size = 4*std(fourth.(res))/sqrt(nrep))
end

function average_errors_one_time_horizon_format_res(forecast_time, nrep, wms_start, Λ_start, α, nparts, errors_fun)
    println("forecast_time = $(round(forecast_time, digits = 4))")
    average_errors_one_time_horizon(forecast_time, nrep, wms_start, Λ_start, α, nparts, errors_fun) |>
        df -> @transform(df, :t = forecast_time, :nparts = nparts, :α = first(α))
end

function average_log_L2_error_table_one_method(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, log_L2_error_fun)
    expand_grid(time = time_grid, nparts = nparts_grid)        |>
        df -> expected_log_L2_error_one_time_horizon_format_res.(df[!, :time], nrep, Ref(wms_start), δ, θ, γ, σ, Ref(Λ_start), df[!, :nparts], Ref(log_L2_error_fun)) |>
        dflist -> vcat(dflist...)
end

function average_errors_table_one_method(time_grid, nrep, wms_start, Λ_start, α, nparts_grid, errors_fun)
    expand_grid(time = time_grid, nparts = nparts_grid)        |>
    df -> average_errors_one_time_horizon_format_res.(df[!, :time], nrep, Ref(wms_start), Ref(Λ_start), Ref(α), df[!, :nparts], Ref(errors_fun)) |>
        dflist -> vcat(dflist...)
end

function average_log_L2_error_table_full_table(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, log_L2_error_fun_dict)
    method_names = keys(log_L2_error_fun_dict) |> collect

    map(method_name -> average_log_L2_error_table_one_method(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, log_L2_error_fun_dict[method_name]) |> 
        df -> @transform(df, :Method = method_name), method_names) |>
        dflist -> vcat(dflist...)

end

function average_errors_table_full_table(time_grid, nrep, wms_start, Λ_start, α, nparts_grid, errors_fun_dict)

    method_names = keys(errors_fun_dict) |> collect

    map(method_name -> average_errors_table_one_method(time_grid, nrep, wms_start, Λ_start, α, nparts_grid, errors_fun_dict[method_name]) |> 
        df -> @transform(df, :Method = method_name), method_names) |>
        dflist -> vcat(dflist...)

end