include("expand_grid.jl")

function compute_log_L2_error_CIR_particle_BD(wms_ref, α_ref, β_ref, wms_start, δ, θ, γ, σ, Λ_start, forecast_time; nparts=100)
    particle_BD = DualOptimalFiltering.CIR_particle_BD_prediction_step_tavare(wms_start, Λ_start, forecast_time, δ, θ, γ, σ; nparts=nparts)

    Λ_BD = keys(particle_BD)
    log_wms_BD = log.(getindex.(Ref(particle_BD), Λ_BD))

    α_BD, β_BD = DualOptimalFiltering.create_gamma_mixture_parameters(δ, θ, Λ_BD)

    logdist = DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.(wms_ref), α_ref, β_ref, log_wms_BD, α_BD, β_BD)
end

function compute_errors_mixtures(wms_1, α_1, β_1, wms_2, α_2, β_2)
    logdist = DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.(wms_1), α_1, β_1, log.(wms_2), α_2, β_2)
    first_moment_1 = sum(wms_1 .* α_1 ./ β_1)
    first_moment_2 = sum(wms_2 .* α_2 ./ β_2)
    second_moment_1 = sum(wms_1 .* α_1 .* (α_1.+ 1) ./ (β_1 .^ 2))
    second_moment_2 = sum(wms_2 .* α_2 .* (α_2 .+ 1) ./ (β_2 .^ 2))
    first_moment_abs_error = abs(first_moment_1 - first_moment_2)
    second_moment_abs_error = abs(second_moment_1 - second_moment_2)
    sd_abs_error = abs(sqrt(second_moment_1 - first_moment_1^2) - sqrt(second_moment_2 - first_moment_2^2))

    return logdist, first_moment_abs_error, second_moment_abs_error, sd_abs_error
end

function compute_errors_CIR_particle_BD(wms_ref, α_ref, β_ref, wms_start, δ, θ, γ, σ, Λ_start, forecast_time; nparts=100)

    particle_BD = DualOptimalFiltering.CIR_particle_BD_prediction_step_tavare(wms_start, Λ_start, forecast_time, δ, θ, γ, σ; nparts=nparts)

    Λ_BD = keys(particle_BD)
    wms_BD = getindex.(Ref(particle_BD), Λ_BD)

    α_BD, β_BD = DualOptimalFiltering.create_gamma_mixture_parameters(δ, θ, Λ_BD)

    return compute_errors_mixtures(wms_ref, α_ref, β_ref, wms_BD, α_BD, β_BD)

end


function compute_log_L2_error_CIR_particle_PD(wms_ref, α_ref, β_ref, wms_start, δ, θ, γ, σ, Λ_start, forecast_time; nparts=100)
    particle_PD = DualOptimalFiltering.CIR_particle_integrated_PD_prediction_step(wms_start, Λ_start, forecast_time, θ, γ, σ; nparts=nparts)

    p = γ/σ^2*1/(θ*exp(2*γ*forecast_time) + γ/σ^2 - θ)
    θ_new = p * θ * exp(2*γ*forecast_time)

    Λ_PD = keys(particle_PD)
    log_wms_PD = log.(getindex.(Ref(particle_PD), Λ_PD))

    α_PD, β_PD = DualOptimalFiltering.create_gamma_mixture_parameters(δ, θ_new, Λ_PD)


    logdist = DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.(wms_ref), α_ref, β_ref, log_wms_PD, α_PD, β_PD)
end

function compute_errors_CIR_particle_PD(wms_ref, α_ref, β_ref, wms_start, δ, θ, γ, σ, Λ_start, forecast_time; nparts=100)
    particle_PD = DualOptimalFiltering.CIR_particle_integrated_PD_prediction_step(wms_start, Λ_start, forecast_time, θ, γ, σ; nparts=nparts)

    p = γ/σ^2*1/(θ*exp(2*γ*forecast_time) + γ/σ^2 - θ)
    θ_new = p * θ * exp(2*γ*forecast_time)

    Λ_PD = keys(particle_PD)
    wms_PD = getindex.(Ref(particle_PD), Λ_PD)

    α_PD, β_PD = DualOptimalFiltering.create_gamma_mixture_parameters(δ, θ_new, Λ_PD)

    return compute_errors_mixtures(wms_ref, α_ref, β_ref, wms_PD, α_PD, β_PD)

end

function compute_log_L2_error_CIR_PF(wms_ref, α_ref, β_ref, wms_start, δ, θ, γ, σ, Λ_start, forecast_time; nparts=100)
    particle_PF_approx = DualOptimalFiltering.CIR_particle_PF_approx_prediction_step(wms_start, Λ_start, forecast_time, δ, θ, γ, σ; nparts=nparts)

    α_PF, β_PF = DualOptimalFiltering.create_gamma_kde_mixture_parameters(particle_PF_approx)

    logdist = DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.(wms_ref), α_ref, β_ref, repeat([1. / length(α_PF)], inner = length(α_PF)), α_PF, β_PF)
end

function compute_errors_CIR_PF(wms_ref, α_ref, β_ref, wms_start, δ, θ, γ, σ, Λ_start, forecast_time; nparts=100)
    particle_PF_approx = DualOptimalFiltering.CIR_particle_PF_approx_prediction_step(wms_start, Λ_start, forecast_time, δ, θ, γ, σ; nparts=nparts)

    α_PF, β_PF = DualOptimalFiltering.create_gamma_kde_mixture_parameters(particle_PF_approx)

    logdist = DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.(wms_ref), α_ref, β_ref, repeat([1. / length(α_PF)], inner = length(α_PF)), α_PF, β_PF)

    first_moment_ref = sum(wms_ref .* α_ref ./ β_ref)
    second_moment_ref = sum(wms_ref .* α_ref .* (α_ref .+ 1) ./ (β_ref .^ 2))
    sd_ref = sqrt(second_moment_ref - first_moment_ref^2)

    first_moment_PF = mean(particle_PF_approx)
    second_moment_PF = var(particle_PF_approx) + first_moment_PF^2
    sd_PF = std(particle_PF_approx)

    return logdist, abs(first_moment_ref - first_moment_PF), abs(second_moment_ref - second_moment_PF), abs(sd_ref - sd_PF)
end

function average_log_L2_error_one_time_horizon(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, log_L2_error_fun)

    θ_ref, Λ_ref, wms_ref = DualOptimalFiltering.predict_CIR_params(wms_start, 1., θ, γ, σ, Λ_start, forecast_time)

    α_ref, β_ref = DualOptimalFiltering.create_gamma_mixture_parameters(δ, θ_ref, Λ_ref)

    map(i -> log_L2_error_fun(wms_ref, α_ref, β_ref, wms_start, δ, θ, γ, σ, Λ_start, forecast_time; nparts=nparts), 1:nrep) |> v -> DataFrame(expected_log_L2_error = logsumexp(v) - log(nrep), CI_size = 4*std(v)/sqrt(nrep))
    #The log distances look approximately gaussian with a slight asymmetry to the left, the distances look approximately gaussian with a slight asymmetry to the right. So using Gaussian CI on the log

end

function nth(v, n)
    return v[n]
end

second(v) = nth(v, 2)
third(v) = nth(v, 3)
fourth(v) = nth(v, 4)

function average_errors_one_time_horizon(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, errors_fun)

    θ_ref, Λ_ref, wms_ref = DualOptimalFiltering.predict_CIR_params(wms_start, 1., θ, γ, σ, Λ_start, forecast_time)

    α_ref, β_ref = DualOptimalFiltering.create_gamma_mixture_parameters(δ, θ_ref, Λ_ref)

    res = map(i -> errors_fun(wms_ref, α_ref, β_ref, wms_start, δ, θ, γ, σ, Λ_start, forecast_time; nparts=nparts), 1:nrep)

    DataFrame(expected_log_L2_error = logsumexp(first.(res)) - log(nrep), log_L2_CI_size = 4*std(first.(res))/sqrt(nrep),
    expected_first_moment_error = mean(second.(res)), first_moment_CI_size = 4*std(second.(res))/sqrt(nrep),
    expected_second_moment_error = mean(third.(res)), second_moment_CI_size = 4*std(third.(res))/sqrt(nrep), 
    expected_sd_error = mean(fourth.(res)), sd_CI_size = 4*std(fourth.(res))/sqrt(nrep))
end

function expected_log_L2_error_one_time_horizon_format_res(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, log_L2_error_fun)
    average_log_L2_error_one_time_horizon(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, log_L2_error_fun) |>
        df -> @transform(df, :t = forecast_time, :nparts = nparts, :δ = δ, :θ = θ, :γ = γ, :σ = σ)
end

function average_errors_one_time_horizon_format_res(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, errors_fun)
    println("forecast_time = $(round(forecast_time, digits = 2))")
    average_errors_one_time_horizon(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, errors_fun) |>
        df -> @transform(df, :t = forecast_time, :nparts = nparts, :δ = δ, :θ = θ, :γ = γ, :σ = σ)
end

function average_log_L2_error_table_one_method(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, log_L2_error_fun)
    expand_grid(time = time_grid, nparts = nparts_grid)        |>
        df -> expected_log_L2_error_one_time_horizon_format_res.(df[!, :time], nrep, Ref(wms_start), δ, θ, γ, σ, Ref(Λ_start), df[!, :nparts], Ref(log_L2_error_fun)) |>
        dflist -> vcat(dflist...)
end

function average_errors_table_one_method(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, errors_fun)
    expand_grid(time = time_grid, nparts = nparts_grid)        |>
    df -> average_errors_one_time_horizon_format_res.(df[!, :time], nrep, Ref(wms_start), δ, θ, γ, σ, Ref(Λ_start), df[!, :nparts], Ref(errors_fun)) |>
        dflist -> vcat(dflist...)
end

function average_log_L2_error_table_full_table(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, log_L2_error_fun_dict)
    method_names = keys(log_L2_error_fun_dict) |> collect

    map(method_name -> average_log_L2_error_table_one_method(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, log_L2_error_fun_dict[method_name]) |> 
        df -> @transform(df, :Method = method_name), method_names) |>
        dflist -> vcat(dflist...)

end

function average_errors_table_full_table(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, errors_fun_dict)
    method_names = keys(errors_fun_dict) |> collect

    map(method_name -> average_errors_table_one_method(time_grid, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts_grid, errors_fun_dict[method_name]) |> 
        df -> @transform(df, :Method = method_name), method_names) |>
        dflist -> vcat(dflist...)

end