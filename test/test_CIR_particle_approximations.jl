@testset "CIR particle prediction tests" begin
    Random.seed!(0)
    times = range(0, stop = 2, length = 20)
    x0 = 3
    δ = 3.
    γ = 0.5
    σ = 1.
    Δt = 0.1
    λ = 1.
    X = DualOptimalFiltering.generate_CIR_trajectory(times, x0, δ, γ, σ);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(times, Y))
    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR(δ, γ, σ, λ, data);
    θ = θ_of_t[2.]
    wms = wms_of_t[wms_of_t |> keys |> collect |> last]
    Λ = Λ_of_t[wms_of_t |> keys |> collect |> last]
    res = DualOptimalFiltering.CIR_particle_integrated_PD_prediction_step(wms, Λ, Δt, θ, γ, σ)
    ref = DualOptimalFiltering.predict_CIR_params(wms, δ, θ, γ, σ, Λ, Δt)
    # to_plot = R"tibble(lambda = $(ref[2] |> collect), wms = $(ref[3]), type = 'ref') %>% bind_rows(tibble(lambda = $(res |> keys |> collect |> sort), wms = $(getindex.(Ref(res), res |> keys |> collect |> sort)), type = 'res'))"
    #R"$to_plot %>% ggplot(aes(x=lambda, y = wms, colour = type)) + theme_bw() + geom_point() + geom_line()"

    @test length(DualOptimalFiltering.predict_CIR_params_particle_BD_approx(wms, 1., θ, γ, σ, Λ, Δt)) == 3

    @test length(DualOptimalFiltering.CIR_particle_PF_approx_prediction_step(wms, Λ, Δt, δ, θ, γ, σ)) == 1000


    @test DualOptimalFiltering.birth_nu(1, 4) == [0, -1, 1, 0, 0]

    nmax = 3
    res = DualOptimalFiltering.compute_all_rates_BD_CIR(δ, θ, γ, σ, nmax)

    @test size(res[1]) == (4, 6)
    @test length(res[2]) == 6

    # wms = wms_of_t[wms_of_t |> keys |> collect |> last]
    # smp = DualOptimalFiltering.CIR_particle_BD_prediction_step_gillespie(wms, Λ, 0.01, δ, θ, γ, σ, 20; nparts=100)

    # @test smp |> values |> sum == 1.

    @test isfinite(DualOptimalFiltering.g_linear_BD_immigration(1.1, 1.1, 0.5))
    @test isfinite(DualOptimalFiltering.h_linear_BD_immigration(1.1, 1.1, 0.5))

    @test isinteger(DualOptimalFiltering.sim_linear_BD_immigration(5, 0.01, 1.1, 0.8, 0.2))
    @test isinteger(DualOptimalFiltering.sim_linear_BD_immigration(5, 11., 1.1, 0.8, 0.2))

    smp = DualOptimalFiltering.CIR_particle_BD_prediction_step_tavare(wms, Λ, 0.01, δ, θ, γ, σ; nparts=1000)

    @test smp |> values |> sum ≈ 1.


end;

@testset "CIR filtering with particle prediction tests" begin
    Random.seed!(0)
    X = DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = 2, length = 20), 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(range(0, stop = 2, length = 20), Y))
    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_particle_integrated_PD_approx(3., 0.5, 1.,1.,data);
    [@test isreal(k) for k in Λ_of_t |> keys]
    [@test isinteger(sum(k)) for k in Λ_of_t |> values]
    [@test isreal(sum(k)) for k in wms_of_t |> values]
    [@test isreal(k) for k in θ_of_t |> values]

    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_particle_integrated_BD_approx(3., 0.5, 1.,1.,data);
    [@test isreal(k) for k in Λ_of_t |> keys]
    [@test isinteger(sum(k)) for k in Λ_of_t |> values]
    [@test isreal(sum(k)) for k in wms_of_t |> values]
    [@test isreal(k) for k in θ_of_t |> values]
end;