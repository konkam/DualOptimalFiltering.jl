using ExactWrightFisher

@testset "test exact L2 Gamma distance functions" begin
    # @test_warn AssertionError DualOptimalFiltering.log_int_prod_2_Gammas(0.5, 0.5, 0.5, 1)
    @test DualOptimalFiltering.log_int_prod_2_Gammas(1, 1, 1, 1) == -log(2)
    @test DualOptimalFiltering.log_int_prod_2_Gammas(2, 0.5, 1, 0.5) == log(0.5^2*0.5 )
    # @test DualOptimalFiltering.log_int_prod_2_Gammas(0.1, 1.2, 1.1, 0.3) ≈ Float64(log(DualOptimalFiltering.prod_2_gammas_arb(0.1, 1.2, 1.1, 0.3) ))

    @test DualOptimalFiltering.int_prod_2_Gammas(1, 1, 1, 1) == 0.5
    @test DualOptimalFiltering.int_prod_2_Gammas(2, 0.5, 1, 0.5) ≈ 0.5^2*0.5



    @test_nowarn DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.([0.2,0.3,0.5]), [1,2,3], [1,2,4], log.([0.6,0.4]), [1,2], [4,2])
    @test !isfinite(DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.([0.2,0.3,0.5]), [1,2,3], [1,2,4], log.([0.2,0.3,0.5]), [1,2,3], [1,2,4]))

    # @test DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.([0.2,0.3,0.5]), [1,2,3], [1,2,4], log.([0.6,0.4]), [1,2], [4,2]) ≈ Float64(log(DualOptimalFiltering.L2_dist_Gamma_mixtures_arb([0.2,0.3,0.5], [1,2,3], [1,2,4], [0.6,0.4], [1,2], [4,2])))

    Random.seed!(0)
    δ = 3.
    γ = 3.
    σ = 0.5
    λ = 1.
    θ = γ/σ^2
    X = DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = 2, length = 20), δ, γ, σ, λ);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(range(0, stop = 2, length = 20), Y))
    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR(δ, γ, σ, λ, data);
    Λ_of_t1, wms_of_t1, θ_of_t1 = DualOptimalFiltering.filter_CIR_keep_fixed_number(δ, γ, σ, λ, data, 1);

    time_grid = data |> keys |> collect |> sort
    logw1 = log.(wms_of_t[time_grid[20]])
    αlist_1, βlist_1 = DualOptimalFiltering.create_gamma_mixture_parameters(δ, θ_of_t[time_grid[20]], Λ_of_t[time_grid[20]][isfinite.(logw1)])

    logw2 = log.(wms_of_t1[time_grid[20]])
    αlist_2, βlist_2 = DualOptimalFiltering.create_gamma_mixture_parameters(δ, θ_of_t1[time_grid[20]], Λ_of_t1[time_grid[20]])

    @test_nowarn DualOptimalFiltering.log_L2_dist_Gamma_mixtures(logw1[isfinite.(logw1)], αlist_1, βlist_1, logw2, αlist_2, βlist_2)

    cc = DualOptimalFiltering.create_gamma_mixture_pdf(δ, θ_of_t[time_grid[20]], Λ_of_t[time_grid[20]][isfinite.(logw1)], wms_of_t[time_grid[20]][isfinite.(logw1)])

    cc1 = DualOptimalFiltering.create_gamma_mixture_pdf(δ, θ_of_t1[time_grid[20]], Λ_of_t1[time_grid[20]], wms_of_t1[time_grid[20]])

    # res = DualOptimalFiltering.L2_dist_1D(cc, cc1)
    # @test exp(DualOptimalFiltering.log_L2_dist_Gamma_mixtures(logw1[isfinite.(logw1)], αlist_1, βlist_1, logw2, αlist_2, βlist_2)) ≈ res[1] atol=res[2]
end;

@testset "test exact L2 distance functions for Dirichlet mixtures" begin

    @test DualOptimalFiltering.lmvbeta([1,1]) == 0.0
    @test DualOptimalFiltering.log_int_prod_2_Dir([1,1], [1,1]) == 0.0
    @test DualOptimalFiltering.log_int_prod_2_Dir([1,1,1], [1,1,1]) ≈ 0.6931471805599453

    @test_nowarn DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.6,0.4]), [[1,2,1],[4,2,3]])
    @test_nowarn DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]])

    Random.seed!(0);
    α_vec = [1.2, 1.4, 1.3]
    # α = α_vec
    K = length(α_vec)
    Pop_size_WF3 = 10
    Nparts = 100
    time_grid_WF3 = range(0, stop = 1, length = 5)
    wfchain = Wright_Fisher_K_dim_exact_trajectory([0.2, 0.4, 0.4], time_grid_WF3, α_vec)
    wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain[:,k])) for k in 1:size(wfchain,2)] |> l -> hcat(l...)
    data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t]' |> collect for t in 1:size(wfobs_WF3,2)]))

    # log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb = DualOptimalFiltering.precompute_log_terms_arb(data_WF3, sum(α_vec); digits_after_comma_for_time_precision = 4)
    #
    #
    # Λ_of_t_arb, wms_of_t_arb = DualOptimalFiltering.filter_WF_precomputed(α_vec, data_WF3, log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb)

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_number(α_vec, data_WF3, 30)

    times = keys(Λ_of_t) |> collect |> sort


    # res = DualOptimalFiltering.compute_L2_distance_between_two_Dirichlet_mixtures(α_vec, Λ_of_t_arb[times[5]], wms_of_t_arb[times[5]], Λ_of_t[times[5]], wms_of_t[times[5]])
    @test_nowarn DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.(wms_of_t[times[5]]), DualOptimalFiltering.create_dirichlet_mixture(α_vec, Λ_of_t[times[5]]), log.(wms_of_t[times[4]]),  DualOptimalFiltering.create_dirichlet_mixture(α_vec, Λ_of_t[times[4]]))

    # @test_nowarn DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(α_vec, Λ_of_t[times[2]], wms_of_t[times[2]], Λ_of_t[times[3]], wms_of_t[times[3]])

    # @test exp(DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.(wms_of_t_arb[times[3]]), DualOptimalFiltering.create_dirichlet_mixture(α_vec, Λ_of_t_arb[times[3]]), log.(wms_of_t[times[3]]),  DualOptimalFiltering.create_dirichlet_mixture(α_vec, Λ_of_t[times[3]]))) ≈ res[1] atol=res[2]


end;
