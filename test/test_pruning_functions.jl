
@testset "Pruning functions tests" begin
    Random.seed!(0)
    X = DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = 2, length = 20), 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(range(0, stop = 2, length = 20), Y))
    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR(3., 0.5, 1.,1.,data);

    times = Λ_of_t |> keys |> collect |> sort
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_above_threshold(Λ_of_t[times[end]], wms_of_t[times[end]], 0.001)
    @test minimum(wms_of_t_kept) >= 0.001
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 200)
    @test length(wms_of_t_kept |> unique) <= 200
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 2)
    @test length(wms_of_t_kept |> unique) <= 2
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 1)
    @test length(wms_of_t_kept |> unique) <= 1
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_fraction(Λ_of_t[times[end]], wms_of_t[times[end]], 0.95)
    @test sum(wms_of_t_kept) >= 0.95
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_fraction(Λ_of_t[times[end]], wms_of_t[times[end]], 0.99)
    @test sum(wms_of_t_kept) >= 0.99

end;
