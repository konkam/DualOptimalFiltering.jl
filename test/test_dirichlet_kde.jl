using Distributions, Random

@testset "test kernel function" begin
    @test DualOptimalFiltering.dirichletkernel_oneval([0.2,0.3,0.5], [0.16,  0.64,  0.2], 1.2) ≈ 2.3904148155639433 atol=10.0^(-5)
    @test DualOptimalFiltering.dirichletkernel_oneval([0.0,0.5,0.5], [0.16,  0.64,  0.2], 1.2) == 0
    # Random.seed!(1)
    X  = rand(Dirichlet(ones(4)),10)'
    w = ones(size(X, 1))
    @test_nowarn dirichletkernel([0.2,0.3,0.4, 0.1], X, 1.2, w, size(X, 1))
    # res = [7.85877, 7.34332, 8.70397, 8.05658, 4.97756, 8.25494, 8.63194, 7.72478, 7.36807, 8.70078]
    # for i in eachindex(w)
    #     @test w[i] .≈ res[i] atol=10.0^(-5)
    # end
    X_list = [X[i,:] for i in 1:size(X,1)]
    @test_nowarn  dirichletkernel([0.2,0.3,0.4, 0.1], X, 1.2, w, size(X, 1))
    # for i in eachindex(w)
    #     @test w[i] .≈ res[i] atol=10.0^(-5)
    # end

    @test DualOptimalFiltering.dirichletkernel_marginals_oneval(1, 1, [1,1,1,1], 1; log = false) == 0


    @test length(DualOptimalFiltering.dirichletkernel_marginals_oneval([1,1,1,1], [1,1,1,1], 1; log = false)) == 4

    tt = DualOptimalFiltering.dirichletkernel_marginals_oneval([0.8,0.8,0.8,0.8], [0,0,0,0], 1; log = false) .== pdf(Beta(1, 3), 0.8)

    for t in tt
        @test t
    end

    @test length(DualOptimalFiltering.dirichletkernel_marginals_oneval([1,1,1,1], [1,1,1,1], 1; log = false)) == 4


    @test DualOptimalFiltering.dirichletkernel_marginals(1, 1, X, 1; log = false) == 0

    tt = DualOptimalFiltering.dirichletkernel_marginals([1,1,1,1], collect(X), 1; log = false) .== 0

    for t in tt
        @test t
    end

end;


@testset "test bandwidth selection functions" begin
    Random.seed!(1)
    X  = rand(Dirichlet(ones(4)),10)'
    @test DualOptimalFiltering.midrange([0,1]) ≈ 0.5
    @test_nowarn DualOptimalFiltering.lcv(X, dirichletkernel, DualOptimalFiltering.midrange(X), ones(size(X, 1)), size(X, 1))
    @test_nowarn DualOptimalFiltering.bwlcv(X, dirichletkernel)
    @test_nowarn DualOptimalFiltering.bwlcv_large_bounds(X, dirichletkernel)
    @test_nowarn DualOptimalFiltering.minus_log_leaveoneout(X, dirichletkernel, DualOptimalFiltering.midrange(X), ones(size(X, 1)), size(X, 1))
    @test_nowarn DualOptimalFiltering.bwloo_large_bounds(X, dirichletkernel)
    X_list = [X[i,:] for i in 1:size(X,1)]
    @test_nowarn DualOptimalFiltering.midrange(X_list)
    @test_nowarn  DualOptimalFiltering.lcv(X_list, dirichletkernel, DualOptimalFiltering.midrange(X_list), ones(size(X, 1)), size(X, 1))
    @test_nowarn DualOptimalFiltering.bwlcv(X_list, dirichletkernel) 
end;
