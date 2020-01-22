@testset "update CIR Tests" begin
    tmp = DualOptimalFiltering.update_CIR_params([0.5, 0.5], 1., 1.5, 1., [0, 1], [5])
    @test tmp[1] == 2.5
    @test tmp[2] == [5, 6]
    @test tmp[3][1] ≈ 0.131579 atol=10.0^(-5)
    @test tmp[3][2] ≈ 0.868421 atol=10.0^(-5)
end;

@testset "predict CIR Tests" begin
    tmp = DualOptimalFiltering.predict_CIR_params([1.], 1., 2., 1., 1., [5], 1.)
    @test tmp[1] ≈ 1.072578883495754 atol=10.0^(-5)
    @test tmp[2] == 0:5
    [@test tmp[3][k] ≈ [0.686096, 0.268465, 0.0420196, 0.0032884, 0.000128673, 2.01396e-6][k] atol=10.0^(-5) for k in 1:6]
end

@testset "CIR filtering tests" begin
    Random.seed!(0)
    X = DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = 2, length = 20), 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(range(0, stop = 2, length = 20), Y))
    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR(3., 0.5, 1.,1.,data);
    [@test isreal(k) for k in Λ_of_t |> keys]
    [@test isinteger(sum(k)) for k in Λ_of_t |> values]
    [@test isreal(sum(k)) for k in wms_of_t |> values]
    [@test isreal(k) for k in θ_of_t |> values]
end;
