@testset "CIR approximate filtering tests" begin
    Random.seed!(4)

    X = DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = 2, length = 20), 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(range(0, stop = 2, length = 20), Y))


    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_keep_fixed_number(3., 0.5, 1.,1., data, 10);

    @test length(keys(Λ_of_t)) == 20
    @test length(keys(wms_of_t)) == 20

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_CIR_keep_above_threshold(3., 0.5, 1.,1., data, 0.0001)
    @test length(keys(Λ_of_t)) == 20
    @test length(keys(wms_of_t)) == 20

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_CIR_fixed_fraction(3., 0.5, 1.,1., data, .99)
    @test length(keys(Λ_of_t)) == 20
    @test length(keys(wms_of_t)) == 20


    fixed_number = 10
    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
        return Λ_of_t_kept, DualOptimalFiltering.normalise(wms_of_t_kept)
    end

end;
