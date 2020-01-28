@testset "test full smoothing CIR" begin

    Random.seed!(0)
    times_sim = range(0, stop = 20, length = 20)
    X = DualOptimalFiltering.generate_CIR_trajectory(times_sim, 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),1), X);
    data = Dict(zip(range(0, stop = 2, length = 20), Y))
    δ, γ, σ, λ = 3., 0.5, 1., 1.
    Λ_of_t, logwms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_logweights(δ, γ, σ, λ, data);

    Random.seed!(0)
    @test_nowarn DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, 1, 1, 1, data)

end

@testset "stringent test for full smoothing CIR" begin

    function simulate_CIR_data(;Nsteps_CIR = 200)
        Random.seed!(2)

        δ = 3.
        γ = 2.5
        σ = 2.
        Nobs = 2
        dt_CIR = 0.011
        # max_time = .1
        # max_time = 0.001
        λ = 1.

        time_grid_CIR = [k*dt_CIR for k in 0:(Nsteps_CIR-1)]
        X_CIR = DualOptimalFiltering.generate_CIR_trajectory(time_grid_CIR, 3, δ, γ, σ)
        Y_CIR = map(λ -> rand(Poisson(λ), Nobs), X_CIR);
        data_CIR = Dict(zip(time_grid_CIR, Y_CIR))
        return data_CIR, Y_CIR, X_CIR, time_grid_CIR, δ, γ, σ, λ
    end

    data_CIR, Y_CIR, X_CIR, times, δ, γ, σ, λ = simulate_CIR_data(;Nsteps_CIR = 800)

    θ_it_δγ_param = (δ, γ, σ)


    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_keep_fixed_number(δ, γ, σ, λ, data_CIR, 10; silence = false)

    fixed_number = 50

    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
        return Λ_of_t_kept, DualOptimalFiltering.normalise(wms_of_t_kept)
    end


    Λ_of_t_pruned, wms_of_t_pruned =  DualOptimalFiltering.prune_all_dicts(Λ_of_t, wms_of_t, prune_keeping_fixed_number)


    @test_nowarn DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(θ_it_δγ_param[1], θ_it_δγ_param[2], θ_it_δγ_param[3], Λ_of_t_pruned, wms_of_t_pruned  |> DualOptimalFiltering.convert_weights_to_logweights, θ_of_t, 1, 1, 1, data_CIR)

end;
