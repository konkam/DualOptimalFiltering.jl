include("variability_of_prediction_WF.jl")

function generate_WF_data(α = [1.1, 1.1, 1.1], Ntimes_WF3 = 4, maxt = 20, X0 = rand(Dirichlet(3, 0.3)))

    K = length(α)

    Pop_size_WF3 = 20
    time_step_WF3 = maxt/Ntimes_WF3
    time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
    wfchain_WF3 = Wright_Fisher_K_dim_trajectory_with_t005_approx(X0, time_grid_WF3[1:(end-1)], α)
    wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
    data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))
    return data_WF3, wfchain_WF3
end



function error_one_time_mixtures(Λ_ref, wms_ref, Λ_approx, wms_approx, α)

    α_ref = DualOptimalFiltering.create_dirichlet_mixture(α, Λ_ref)
    α_approx = DualOptimalFiltering.create_dirichlet_mixture(α, Λ_approx)


    compute_moment_errors_Dirichlet_mixtures(wms_ref[wms_ref .> 0], α_ref[wms_ref .> 0], wms_approx[wms_approx .> 0], α_approx[wms_approx .> 0])
end

function error_signal_retrieval_multidim_one_time_mixture(X, Λ, wms, α)

    @assert length(α) == length(X)

    αs = DualOptimalFiltering.create_dirichlet_mixture(α, Λ)

    marginal_first_moments_1 = reduce(add_local, (wms .* (α_i ./ sum(α_i) for α_i in αs)))

    return abs.(X - marginal_first_moments_1) |> mean #mean instead of sum should allow comparing over various values of K
end

function median_error_signal_retrieval_mixture(X, Λ_of_t, wms_of_t, α)

    #Here assuming that X is a KxNtimes matrix
    @assert length(α) == size(X, 1)

    times = Λ_of_t |> keys |> collect |> sort

    return Float64[error_signal_retrieval_multidim_one_time_mixture(X[:,i], Λ_of_t[times[i]], wms_of_t[times[i]], α)  for i in eachindex(times)] |> median
end


function median_error_over_time_mixtures(Λ_of_t_ref, wms_of_t_ref, Λ_of_t_approx, wms_of_t_approx, α)
    times = Λ_of_t_ref |> keys |> collect |> sort
    map(t-> error_one_time_mixtures(Λ_of_t_ref[t], wms_of_t_ref[t], Λ_of_t_approx[t], wms_of_t_approx[t], α), times) |>
        l -> [median(first.(l)),
        median(second.(l)),
        median(third.(l)),
        median(fourth.(l))]

end

function expected_log_L2_error_one_time_horizon_format_res(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, log_L2_error_fun)
    average_log_L2_error_one_time_horizon(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, log_L2_error_fun) |>
        df -> @transform(df, :t = forecast_time, :nparts = nparts, :δ = δ, :θ = θ, :γ = γ, :σ = σ)
end

function local_square(x)
    return x .* x
end

function compute_marginal_filtering_moments_from_pf_multidim(pf)
    pf_length = pf["resampled"] |> length

    first_moments = Vector{Float64}[sum(pf["X"][i] .* exp.(pf["logW"][:,i])) for i in 1:pf_length]
    second_moments = Vector{Float64}[sum( (local_square.(pf["X"][i])) .* exp.(pf["logW"][:,i])) for i in 1:pf_length]
    return first_moments, second_moments
end

function compute_marginal_filtering_moments_from_mixture(Λ_of_t, wms_of_t, α)

    times = Λ_of_t |> keys |> collect |> sort

    marginal_first_moments = Vector{Float64}[α .* 0. for k in eachindex(times)]
    marginal_second_moments = Vector{Float64}[α .* 0. for k in eachindex(times)]

    for t in 1:length(times)

        α_t = DualOptimalFiltering.create_dirichlet_mixture(α, Λ_of_t[times[t]])

        wms_t = wms_of_t[times[t]]

        marginal_first_moments[t] = reduce(add_local, (wms_t .* (α_i ./ sum(α_i) for α_i in α_t)))
        marginal_second_moments[t] = reduce(add_local, (wms_t .* (α_i .* (α_i .+ 1) ./ (sum(α_i)*(sum(α_i)+1)) for α_i in α_t)))
    end

    return marginal_first_moments, marginal_second_moments
end


function median_error_over_time_PF(Λ_of_t_ref, wms_of_t_ref, pf_multidim, α)

    marginal_first_moments_ref, marginal_second_moments_ref = compute_marginal_filtering_moments_from_mixture(Λ_of_t_ref, wms_of_t_ref, α)

    marginal_sds_ref = map((x,y) -> sqrt.(x .- y .^ 2), marginal_second_moments_ref, marginal_first_moments_ref)


    marginal_first_moments_pf, marginal_second_moments_pf = compute_marginal_filtering_moments_from_pf_multidim(pf_multidim)

    marginal_sds_pf = map((x,y) -> sqrt.(x .- y .^ 2), marginal_second_moments_pf, marginal_first_moments_pf)


    median_first_moment_error = map((x,y) -> mean(abs.(x .- y)), marginal_first_moments_ref, marginal_first_moments_pf) |> median
    median_second_moment_error = map((x,y) -> mean(abs.(x .- y)), marginal_second_moments_ref, marginal_second_moments_pf) |> median
    median_sd_error = map((x,y) -> mean(abs.(x .- y)), marginal_sds_ref, marginal_sds_pf) |> median

    return median_first_moment_error, median_second_moment_error, median_sd_error

end

function median_error_multidom_signal_retrieval_PF(X, pf_multidim)
    marginal_first_moments_pf, = compute_marginal_filtering_moments_from_pf_multidim(pf_multidim)

    [mean(abs.(marginal_first_moments_pf[i] .- X[:,i])) for i in eachindex(marginal_first_moments_pf)] |> median

end

function WF_particle_PD_gillespie_prediction(wms, Λ, Δt, α, sα; nparts = 1000)
    DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step(wms, sα, Λ, Δt; nparts = nparts)
end


function WF_particle_PD_integrated_prediction(wms, Λ, Δt, α, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; nparts = 1000)
    DualOptimalFiltering.WF_particle_integrated_PD_prediction_step(wms, sα, Λ, Δt, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; nparts = nparts)
end

function WF_particle_BD_Moran_gillespie_prediction(wms, Λ, Δt, α, sα; nparts = 1000)
    DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_Moran_gillespie(wms, Λ, Δt, α; nparts = nparts)
end

function WF_particle_BD_WF_gillespie_prediction(wms, Λ, Δt, α, sα; nparts = 1000)
    DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_WF_gillespie(wms, Λ, Δt, α; nparts = nparts)
end

function WF_particle_BD_WF_diffusion_prediction(wms, Λ, Δt, α, sα; nparts = 1000)
    DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_WF_diffusion(wms, Λ, Δt, α; nparts = nparts)
end

function compute_errors_one_dataset(data, X, α, Nparts; verbose = false)

    dd = data |> DualOptimalFiltering.prepare_WF_dat_1D_2D |> last

    if verbose
        println("Starting exact computation")
    end

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_number(α, dd, max(10000, 10*Nparts) ; silence=!verbose, trim0=true);

    if verbose
        println("Starting Bootstrap Particle Filtering")
    end

    pf = DualOptimalFiltering.fit_particle_filter_WF(data, α, Nparts = Nparts)

    pf_error = median_error_over_time_PF(Λ_of_t, wms_of_t, pf, α) |> collect


    if verbose
        println("Starting PD_Gillespie")
    end

    Λ_of_t_part_PD_Gillespie, wms_of_t_part_PD_Gillespie = DualOptimalFiltering.filter_WF_particle_approx(α, dd, WF_particle_PD_gillespie_prediction; silence=true, trim0=true, nparts = Nparts)


    PD_Gillespie = median_error_over_time_mixtures(Λ_of_t, wms_of_t, Λ_of_t_part_PD_Gillespie, wms_of_t_part_PD_Gillespie, α)


    if verbose
        println("Starting PD_integrated")
    end

    Λ_of_t_part_PD_integrated, wms_of_t_part_PD_integrated = DualOptimalFiltering.filter_WF_particle_approx_adaptive_precomputation_ar(α, dd, WF_particle_PD_integrated_prediction; silence=true, trim0=true, nparts = Nparts)

    PD_integrated = median_error_over_time_mixtures(Λ_of_t, wms_of_t, Λ_of_t_part_PD_integrated, wms_of_t_part_PD_integrated, α)

    if verbose
        println("Starting Moran_Gillespie")
    end

    Λ_of_t_part_BD_Moran_Gillespie, wms_of_t_part_BD_Moran_Gillespie = DualOptimalFiltering.filter_WF_particle_approx(α, dd, WF_particle_BD_Moran_gillespie_prediction; silence=true, trim0=true, nparts = Nparts)

    BD_Moran_Gillespie = median_error_over_time_mixtures(Λ_of_t, wms_of_t, Λ_of_t_part_BD_Moran_Gillespie, wms_of_t_part_BD_Moran_Gillespie, α)

    if verbose
        println("Starting WF_Gillespie")
    end


    Λ_of_t_part_BD_WF_Gillespie, wms_of_t_part_BD_WF_Gillespie = DualOptimalFiltering.filter_WF_particle_approx(α, dd, WF_particle_BD_WF_gillespie_prediction; silence=true, trim0=true, nparts = Nparts)

    BD_WF_Gillespie = median_error_over_time_mixtures(Λ_of_t, wms_of_t, Λ_of_t_part_BD_WF_Gillespie, wms_of_t_part_BD_WF_Gillespie, α)

    if verbose
        println("Starting WF_diffusion")
    end

    Λ_of_t_part_BD_WF_diffusion, wms_of_t_part_BD_WF_diffusion = DualOptimalFiltering.filter_WF_particle_approx(α, dd, WF_particle_BD_WF_diffusion_prediction; silence=true, trim0=true, nparts = Nparts)

    BD_WF_diffusion = median_error_over_time_mixtures(Λ_of_t, wms_of_t, Λ_of_t_part_BD_WF_diffusion, wms_of_t_part_BD_WF_diffusion, α)

    if verbose
        println("Starting pruning")
    end

    Λ_of_t_pruning_fixed_number, wms_of_t_pruning_fixed_number = DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_number(α, dd, 1; silence=true)

    pruning_PD_error = median_error_over_time_mixtures(Λ_of_t, wms_of_t, Λ_of_t_pruning_fixed_number, wms_of_t_pruning_fixed_number, α)


    return Dict{String, Vector{Float64}}(
        "PF" => [NaN; pf_error; median_error_multidom_signal_retrieval_PF(X, pf)], 
        "PD_Gillespie" => [PD_Gillespie; median_error_signal_retrieval_mixture(X, Λ_of_t_part_PD_Gillespie, wms_of_t_part_PD_Gillespie, α)], 
        "PD_integrated" => [PD_integrated; median_error_signal_retrieval_mixture(X, Λ_of_t_part_PD_integrated, wms_of_t_part_PD_integrated, α)], 
        "BD_Moran_Gillespie" => [BD_Moran_Gillespie; median_error_signal_retrieval_mixture(X, Λ_of_t_part_BD_Moran_Gillespie, wms_of_t_part_BD_Moran_Gillespie, α)], 
        "BD_WF_Gillespie" => [BD_WF_Gillespie; median_error_signal_retrieval_mixture(X, Λ_of_t_part_BD_WF_Gillespie, wms_of_t_part_BD_WF_Gillespie, α)], 
        "BD_WF_diffusion" => [BD_WF_diffusion; median_error_signal_retrieval_mixture(X, Λ_of_t_part_BD_WF_diffusion, wms_of_t_part_BD_WF_diffusion, α)], 
        "Pruning (PD)" => [pruning_PD_error; median_error_signal_retrieval_mixture(X, Λ_of_t_pruning_fixed_number, wms_of_t_pruning_fixed_number, α)], 
        "Exact" => [median_error_signal_retrieval_mixture(X, Λ_of_t, wms_of_t, α)])

end

function compute_errors_one_parameter_set(α = repeat([1.1], inner = 3), npoints = 10, maxt = 10, x0 = [0.4, 0.4, 0.2], Nparts = 100; verbose = false)
    data, X = generate_WF_data(α, npoints, maxt, x0)
    compute_errors_one_dataset(data, X, α, Nparts; verbose = verbose)
end

function estimate_errors_one_parameter_set(α = repeat([1.1], inner = 3), npoints = 10, maxt = 10, x0 = [0.4, 0.4, 0.2], Nparts = 100, nrep = 20; verbose = true)

    PD_Gillespie_first_moment_error =  Vector{Float64}(undef, nrep)
    PD_integrated_first_moment_error =  Vector{Float64}(undef, nrep)
    PD_pruning_first_moment_error =  Vector{Float64}(undef, nrep)
    BD_Moran_Gillespie_first_moment_error =  Vector{Float64}(undef, nrep)
    BD_WF_Gillespie_first_moment_error =  Vector{Float64}(undef, nrep)
    BD_WF_diffusion_first_moment_error =  Vector{Float64}(undef, nrep)
    PF_first_moment_error =  Vector{Float64}(undef, nrep)

    PD_Gillespie_second_moment_error =  Vector{Float64}(undef, nrep)
    PD_integrated_second_moment_error =  Vector{Float64}(undef, nrep)
    PD_pruning_second_moment_error =  Vector{Float64}(undef, nrep)
    BD_Moran_Gillespie_second_moment_error =  Vector{Float64}(undef, nrep)
    BD_WF_Gillespie_second_moment_error =  Vector{Float64}(undef, nrep)
    BD_WF_diffusion_second_moment_error =  Vector{Float64}(undef, nrep)
    PF_second_moment_error =  Vector{Float64}(undef, nrep)

    PD_Gillespie_sd_error =  Vector{Float64}(undef, nrep)
    PD_integrated_sd_error =  Vector{Float64}(undef, nrep)
    PD_pruning_sd_error =  Vector{Float64}(undef, nrep)
    BD_Moran_Gillespie_sd_error =  Vector{Float64}(undef, nrep)
    BD_WF_Gillespie_sd_error =  Vector{Float64}(undef, nrep)
    BD_WF_diffusion_sd_error =  Vector{Float64}(undef, nrep)
    PF_sd_error =  Vector{Float64}(undef, nrep)


    PD_Gillespie_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    PD_integrated_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    PD_pruning_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    BD_Moran_Gillespie_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    BD_WF_Gillespie_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    BD_WF_diffusion_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    PF_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    Exact_signal_retrieval_error =  Vector{Float64}(undef, nrep)

    Threads.@threads for i in 1:nrep
        if verbose
            println("α, npoints, maxt, x0, Nparts = $α, $npoints, $maxt, $x0, $Nparts, replicate $i / $nrep")
        end
        tmp = compute_errors_one_parameter_set(α, npoints, maxt, x0, Nparts; verbose = verbose)

        PD_Gillespie_first_moment_error[i] = tmp["PD_Gillespie"][2]
        PD_integrated_first_moment_error[i] =  tmp["PD_integrated"][2]
        PD_pruning_first_moment_error[i] =  tmp["Pruning (PD)"][2]
        BD_Moran_Gillespie_first_moment_error[i] = tmp["BD_Moran_Gillespie"][2]
        BD_WF_Gillespie_first_moment_error[i] = tmp["BD_WF_Gillespie"][2]
        BD_WF_diffusion_first_moment_error[i] = tmp["BD_WF_diffusion"][2]
        PF_first_moment_error[i] = tmp["PF"][2]


        PD_Gillespie_second_moment_error[i] = tmp["PD_Gillespie"][3]
        PD_integrated_second_moment_error[i] =  tmp["PD_integrated"][3]
        PD_pruning_second_moment_error[i] =  tmp["Pruning (PD)"][3]
        BD_Moran_Gillespie_second_moment_error[i] = tmp["BD_Moran_Gillespie"][3]
        BD_WF_Gillespie_second_moment_error[i] = tmp["BD_WF_Gillespie"][3]
        BD_WF_diffusion_second_moment_error[i] = tmp["BD_WF_diffusion"][3]
        PF_second_moment_error[i] = tmp["PF"][3]

        PD_Gillespie_sd_error[i] = tmp["PD_Gillespie"][4]
        PD_integrated_sd_error[i] =  tmp["PD_integrated"][4]
        PD_pruning_sd_error[i] =  tmp["Pruning (PD)"][4]
        BD_Moran_Gillespie_sd_error[i] = tmp["BD_Moran_Gillespie"][4]
        BD_WF_Gillespie_sd_error[i] = tmp["BD_WF_Gillespie"][4]
        BD_WF_diffusion_sd_error[i] = tmp["BD_WF_diffusion"][4]
        PF_sd_error[i] = tmp["PF"][4]

        PD_Gillespie_signal_retrieval_error[i] = tmp["PD_Gillespie"][5]
        PD_integrated_signal_retrieval_error[i] =  tmp["PD_integrated"][5]
        PD_pruning_signal_retrieval_error[i] =  tmp["Pruning (PD)"][5]
        BD_Moran_Gillespie_signal_retrieval_error[i] = tmp["BD_Moran_Gillespie"][5]
        BD_WF_Gillespie_signal_retrieval_error[i] = tmp["BD_WF_Gillespie"][5]
        BD_WF_diffusion_signal_retrieval_error[i] = tmp["BD_WF_diffusion"][5]
        PF_signal_retrieval_error[i] = tmp["PF"][5]
        Exact_signal_retrieval_error[i] = tmp["Exact"][1]
    end

    println("Parallel part was finished")
    
        DataFrame(
            method = ["PD_Gillespie", "PD_integrated", "BD_Moran_Gillespie", "BD_WF_Gillespie", "BD_WF_diffusion", "PF", "Pruning (PD)"],
            expected_first_moment_error = [mean(PD_Gillespie_first_moment_error), 
            mean(PD_integrated_first_moment_error), 
            mean(BD_Moran_Gillespie_first_moment_error), 
            mean(BD_WF_Gillespie_first_moment_error), 
            mean(BD_WF_diffusion_first_moment_error), 
            mean(PF_first_moment_error), 
            mean(PD_pruning_first_moment_error)], 
            first_moment_CI_size = map(v -> 4*std(v)/sqrt(nrep), [PD_Gillespie_first_moment_error,
            PD_integrated_first_moment_error, 
            BD_Moran_Gillespie_first_moment_error, 
            BD_WF_Gillespie_first_moment_error, 
            BD_WF_diffusion_first_moment_error, 
            PF_first_moment_error, 
            PD_pruning_first_moment_error]),
            expected_second_moment_error = [mean(PD_Gillespie_second_moment_error), 
            mean(PD_integrated_second_moment_error), 
            mean(BD_Moran_Gillespie_second_moment_error), 
            mean(BD_WF_Gillespie_second_moment_error), 
            mean(BD_WF_diffusion_second_moment_error), 
            mean(PF_second_moment_error), 
            mean(PD_pruning_second_moment_error)], 
            second_moment_CI_size = map(v -> 4*std(v)/sqrt(nrep), [PD_Gillespie_second_moment_error,
            PD_integrated_second_moment_error, 
            BD_Moran_Gillespie_second_moment_error, 
            BD_WF_Gillespie_second_moment_error, 
            BD_WF_diffusion_second_moment_error, 
            PF_second_moment_error, 
            PD_pruning_second_moment_error]),
            expected_sd_error = [mean(PD_Gillespie_sd_error), 
            mean(PD_integrated_sd_error), 
            mean(BD_Moran_Gillespie_sd_error), 
            mean(BD_WF_Gillespie_sd_error), 
            mean(BD_WF_diffusion_sd_error), 
            mean(PF_sd_error), 
            mean(PD_pruning_sd_error)], 
            sd_CI_size = map(v -> 4*std(v)/sqrt(nrep), [PD_Gillespie_sd_error,
            PD_integrated_sd_error, 
            BD_Moran_Gillespie_sd_error, 
            BD_WF_Gillespie_sd_error, 
            BD_WF_diffusion_sd_error, 
            PF_sd_error, 
            PD_pruning_sd_error]),
            expected_signal_retrieval_error = [mean(PD_Gillespie_signal_retrieval_error), 
            mean(PD_integrated_signal_retrieval_error), 
            mean(BD_Moran_Gillespie_signal_retrieval_error), 
            mean(BD_WF_Gillespie_signal_retrieval_error), 
            mean(BD_WF_diffusion_signal_retrieval_error), 
            mean(PF_signal_retrieval_error), 
            mean(PD_pruning_signal_retrieval_error)], 
            signal_retrieval_CI_size = map(v -> 4*std(v)/sqrt(nrep), [PD_Gillespie_signal_retrieval_error,
            PD_integrated_signal_retrieval_error, 
            BD_Moran_Gillespie_signal_retrieval_error, 
            BD_WF_Gillespie_signal_retrieval_error, 
            BD_WF_diffusion_signal_retrieval_error, 
            PF_signal_retrieval_error, 
            PD_pruning_signal_retrieval_error]),
            ) |>
            df -> vcat(df, DataFrame(method = "Exact", 
                                        expected_first_moment_error = 0,
                                        first_moment_CI_size = 0,
                                        expected_second_moment_error = 0,
                                        second_moment_CI_size = 0,
                                        expected_sd_error = 0,
                                        sd_CI_size = 0,
                                        expected_signal_retrieval_error = mean(Exact_signal_retrieval_error),
                                        signal_retrieval_CI_size = Exact_signal_retrieval_error |> v -> 4*std(v)/sqrt(nrep))) |>
            df -> @transform(df, :alpha = first(α), :Nparts = Nparts, :time_step = maxt/npoints)
end

