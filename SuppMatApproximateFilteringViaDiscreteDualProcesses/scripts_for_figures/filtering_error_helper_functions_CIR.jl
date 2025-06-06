include("variability_of_prediction_CIR.jl")

function generate_CIR_data(δ = 20., npoints = 200, maxt = 20, x0 = 3, γ = 0.5, σ = 1.)
    X = DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = maxt, length = npoints), x0, δ, γ, σ);
    Y = map(λ -> rand(Poisson(λ), 1), X);
    data = Dict(zip(range(0, stop = maxt, length = npoints), Y));
    return data, X
end



function error_one_time_mixtures(Λ_ref, wms_ref, θ_ref, Λ_approx, wms_approx, θ_approx, δ)
    α_ref, β_ref = DualOptimalFiltering.create_Gamma_mixture_parameters(δ, θ_ref, Λ_ref)
    α_approx, β_approx = DualOptimalFiltering.create_Gamma_mixture_parameters(δ, θ_approx, Λ_approx)

    compute_errors_mixtures(wms_ref, α_ref, β_ref, wms_approx, α_approx, β_approx)
end

function error_signal_retrieval_one_time_mixture(X, Λ, wms, θ,  δ)
    α, β = DualOptimalFiltering.create_Gamma_mixture_parameters(δ, θ, Λ)
    first_moment = sum(wms .* α ./ β)
    return abs(X - first_moment)
end

function median_error_signal_retrieval_mixture(X, Λ_of_t, wms_of_t, θ_of_t,  δ)
    times = Λ_of_t |> keys |> collect |> sort
    return Float64[error_signal_retrieval_one_time_mixture(X[i], Λ_of_t[times[i]], wms_of_t[times[i]], θ_of_t[times[i]],  δ)  for i in 1:length(times)] |> median
end


function median_error_signal_retrieval_PF(X, pf)

    first_moments_pf, second_moments_pf = compute_filtering_moments_from_pf(pf)


    return abs.(first_moments_pf .- X) |> median
end

function median_error_over_time_mixtures(Λ_of_t_ref, wms_of_t_ref, θ_of_t_ref, Λ_of_t_approx, wms_of_t_approx, θ_of_t_approx, δ)
    times = Λ_of_t_ref |> keys |> collect |> sort
    map(t-> error_one_time_mixtures(Λ_of_t_ref[t], wms_of_t_ref[t], θ_of_t_ref[t], Λ_of_t_approx[t], wms_of_t_approx[t], θ_of_t_approx[t], δ), times) |>
        l -> [median(first.(l)),
        median(second.(l)),
        median(third.(l)),
        median(fourth.(l))]

end

function expected_log_L2_error_one_time_horizon_format_res(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, log_L2_error_fun)
    average_log_L2_error_one_time_horizon(forecast_time, nrep, wms_start, δ, θ, γ, σ, Λ_start, nparts, log_L2_error_fun) |>
        df -> @transform(df, :t = forecast_time, :nparts = nparts, :δ = δ, :θ = θ, :γ = γ, :σ = σ)
end


function fit_particle_filter(data, δ, γ, σ; Nparts = 100)
    Mt = FeynmanKacParticleFilters.create_transition_kernels_CIR(data, δ, γ, σ)
    logGt = FeynmanKacParticleFilters.create_log_potential_functions_CIR(data)
    RS(W) = rand(Categorical(W), length(W))
    FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)
end

function compute_filtering_moments_from_pf(pf)
    pf_length = pf["resampled"] |> length

    first_moments = Float64[sum(pf["X"][i] .* exp.(pf["logW"][:,i])) for i in 1:pf_length]
    second_moments = Float64[sum( (pf["X"][i].^2) .* exp.(pf["logW"][:,i])) for i in 1:pf_length]
    return first_moments, second_moments
end

function compute_filtering_moments_from_mixture(Λ_of_t, wms_of_t, θ_of_t, δ)

    times = Λ_of_t |> keys |> collect |> sort

    first_moments = Array{Float64,1}(undef, length(times))
    second_moments = Array{Float64,1}(undef, length(times))
    for t in 1:length(times)
        α_ref, β_ref = DualOptimalFiltering.create_Gamma_mixture_parameters(δ, θ_of_t[times[t]], Λ_of_t[times[t]])

        first_moments[t] = sum(wms_of_t[times[t]] .* α_ref ./ β_ref)
        second_moments[t] = sum(wms_of_t[times[t]] .* α_ref .* (α_ref .+ 1) ./ (β_ref .^ 2))
    end

    return first_moments, second_moments
end


function median_error_over_time_PF(Λ_of_t_ref, wms_of_t_ref, θ_of_t_ref, pf, δ)
    times = Λ_of_t_ref |> keys |> collect |> sort

    first_moments_ref, second_moments_ref = compute_filtering_moments_from_mixture(Λ_of_t_ref, wms_of_t_ref, θ_of_t_ref, δ)

    first_moments_pf, second_moments_pf = compute_filtering_moments_from_pf(pf)

    median_first_moment_error = median(abs.(first_moments_ref .- first_moments_pf))
    median_second_moment_error = median(abs.(second_moments_ref .- second_moments_pf))
    median_sd_error = median(abs.(sqrt.(abs.(second_moments_ref .- first_moments_ref.^2)) .- sqrt.(abs.(second_moments_pf - first_moments_pf.^2)))) #abs of second_moments_pf - first_moments_pf.^2 because got numerical error, quick fix

    return median_first_moment_error, median_second_moment_error, median_sd_error

end

function compute_errors_one_dataset(data, X, δ, Nparts, γ, σ)

    #Note that we choose to use a fixed number of particles there. But it might be more efficient to fit the same method with different numbers of particles on the same dataset to have paired comparisons.

    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR(δ, γ, σ, 1., data; silence=true, trim0=true);

    pf = fit_particle_filter(data, δ, γ, σ, Nparts = Nparts)

    pf_error = median_error_over_time_PF(Λ_of_t, wms_of_t, θ_of_t, pf, δ) |> collect


    Λ_of_t_part_PD, wms_of_t_part_PD, θ_of_t_part_PD = DualOptimalFiltering.filter_CIR_particle_integrated_PD_approx(δ, γ, σ, 1., data; silence=true, trim0=true, nparts = Nparts)


    PD_error = median_error_over_time_mixtures(Λ_of_t, wms_of_t, θ_of_t, Λ_of_t_part_PD, wms_of_t_part_PD, θ_of_t_part_PD, δ)

    Λ_of_t_part_BD, wms_of_t_part_BD, θ_of_t_part_BD = DualOptimalFiltering.filter_CIR_particle_integrated_BD_approx(δ, γ, σ, 1., data; silence=true, trim0=true, nparts = Nparts)


    BD_error = median_error_over_time_mixtures(Λ_of_t, wms_of_t, θ_of_t, Λ_of_t_part_BD, wms_of_t_part_BD, θ_of_t_part_BD, δ)

    Λ_of_t_pruning_fixed_number, wms_of_t_pruning_fixed_number, θ_of_t_pruning_fixed_number = DualOptimalFiltering.filter_CIR_keep_fixed_number(δ, γ, σ, 1., data, Nparts; silence=true)

    pruning_PD_error = median_error_over_time_mixtures(Λ_of_t, wms_of_t, θ_of_t, Λ_of_t_pruning_fixed_number, wms_of_t_pruning_fixed_number, θ_of_t_pruning_fixed_number, δ)


    return Dict{String, Vector{Float64}}(
        "PF" => [NaN; pf_error; median_error_signal_retrieval_PF(X, pf)], 
        "PD" => [PD_error; median_error_signal_retrieval_mixture(X, Λ_of_t_part_PD, wms_of_t_part_PD, θ_of_t_part_PD,  δ)], 
        "BD" => [BD_error; median_error_signal_retrieval_mixture(X, Λ_of_t_part_BD, wms_of_t_part_BD, θ_of_t_part_BD,  δ)],
        "Pruning (PD)" => [pruning_PD_error; median_error_signal_retrieval_mixture(X, Λ_of_t_pruning_fixed_number, wms_of_t_pruning_fixed_number, θ_of_t_pruning_fixed_number,  δ)], 
        "Exact" => [median_error_signal_retrieval_mixture(X, Λ_of_t, wms_of_t, θ_of_t,  δ)])

end

function compute_errors_one_parameter_set(δ = 20., npoints = 200, maxt = 20, x0 = 3, γ = 0.5, σ = 1., Nparts = 100)
    data, X = generate_CIR_data(δ, npoints, maxt, x0, γ, σ)
    res = compute_errors_one_dataset(data, X, δ, Nparts, γ, σ)
end

function estimate_errors_one_parameter_set(δ = 20., npoints = 200, maxt = 20, x0 = 3, γ = 0.5, σ = 1., Nparts = 100, nrep = 20)

    PD_first_moment_error =  Vector{Float64}(undef, nrep)
    PD_pruning_first_moment_error =  Vector{Float64}(undef, nrep)
    BD_first_moment_error =  Vector{Float64}(undef, nrep)
    PF_first_moment_error =  Vector{Float64}(undef, nrep)

    PD_second_moment_error =  Vector{Float64}(undef, nrep)
    PD_pruning_second_moment_error =  Vector{Float64}(undef, nrep)
    BD_second_moment_error =  Vector{Float64}(undef, nrep)
    PF_second_moment_error =  Vector{Float64}(undef, nrep)

    PD_sd_error =  Vector{Float64}(undef, nrep)
    PD_pruning_sd_error =  Vector{Float64}(undef, nrep)
    BD_sd_error =  Vector{Float64}(undef, nrep)
    PF_sd_error =  Vector{Float64}(undef, nrep)

    PD_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    PD_pruning_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    BD_signal_retrieval_error =  Vector{Float64}(undef, nrep)
    PF_signal_retrieval_error =  Vector{Float64}(undef, nrep)

    for i in 1:nrep
        tmp = compute_errors_one_parameter_set(δ, npoints, maxt, x0, γ, σ, Nparts)
        PD_first_moment_error[i] = tmp["PD"][2]
        PD_pruning_first_moment_error[i] = tmp["Pruning (PD)"][2]
        BD_first_moment_error[i] = tmp["BD"][2]
        PF_first_moment_error[i] = tmp["PF"][2]

        PD_second_moment_error[i] = tmp["PD"][3]
        PD_pruning_second_moment_error[i] = tmp["Pruning (PD)"][3]
        BD_second_moment_error[i] = tmp["BD"][3]
        PF_second_moment_error[i] = tmp["PF"][3]

        PD_sd_error[i] = tmp["PD"][4]
        PD_pruning_sd_error[i] = tmp["Pruning (PD)"][4]
        BD_sd_error[i] = tmp["BD"][4]
        PF_sd_error[i] = tmp["PF"][4]

        PD_signal_retrieval_error[i] = tmp["PD"][5]
        PD_pruning_signal_retrieval_error[i] = tmp["Pruning (PD)"][5]
        BD_signal_retrieval_error[i] = tmp["BD"][5]
        PF_signal_retrieval_error[i] = tmp["PF"][5]
    end

        DataFrame(
            method = ["PD", "BD", "PF", "Pruning (PD)"],
            expected_first_moment_error = [mean(PD_first_moment_error), mean(BD_first_moment_error), mean(PF_first_moment_error), mean(PD_first_moment_error)], 
            first_moment_CI_size = map(v -> 4*std(v)/sqrt(nrep), [PD_first_moment_error, BD_first_moment_error, PF_first_moment_error, PD_first_moment_error]),
            expected_second_moment_error = [mean(PD_second_moment_error), mean(BD_second_moment_error), mean(PF_second_moment_error), mean(PD_pruning_second_moment_error)], 
            second_moment_CI_size = map(v -> 4*std(v)/sqrt(nrep), [PD_second_moment_error, BD_second_moment_error, PF_second_moment_error, PD_pruning_second_moment_error]),
            expected_sd_error = [mean(PD_sd_error), mean(BD_sd_error), mean(PF_sd_error), mean(PD_pruning_sd_error)], 
            sd_CI_size = map(v -> 4*std(v)/sqrt(nrep), [PD_sd_error, BD_sd_error, PF_sd_error, PD_pruning_sd_error]),
            expected_signal_retrieval_error = [mean(PD_signal_retrieval_error), mean(BD_signal_retrieval_error), mean(PF_signal_retrieval_error), mean(PD_pruning_signal_retrieval_error)],
            signal_retrieval_CI_size =  map(v -> 4*std(v)/sqrt(nrep), [PD_signal_retrieval_error, BD_signal_retrieval_error, PF_signal_retrieval_error, PD_pruning_signal_retrieval_error]),
            ) |>
            df -> @transform(df, :delta = δ, :sigma = σ, :Nparts = Nparts)
end

