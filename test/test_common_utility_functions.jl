using Distributions
@testset "Test common utility functions" begin
    @test DualOptimalFiltering.normalise(1:4) == (1:4)/10

    @test_nowarn DualOptimalFiltering.lgamma_local(50)

    @test DualOptimalFiltering.sample_from_Gamma_mixture(1.2, 0.3, [1,2,3], [0.3, 0.3, 0.4]) |> isreal

    @test length(DualOptimalFiltering.sample_from_Gamma_mixture(1.2, 0.3, [1,2,3], [0.3, 0.3, 0.4], 3)) == 3

    f = DualOptimalFiltering.create_dirichlet_mixture_pdf([1.5, 1.5, 1.5], [[0,0,0], [0,0,0], [0,0,0]], [1/3, 1/3, 1/3])
    @test f([0.1, 0.6, 0.3]) == pdf(Dirichlet([1.5, 1.5, 1.5]), [0.1, 0.6, 0.3])
    f = DualOptimalFiltering.create_dirichlet_mixture_pdf([1.5, 1.5, 1.5], [[1,2,1], [1,2,1], [1,2,1]], [1/3, 1/3, 1/3])
    @test f([0.1, 0.6, 0.3]) == pdf(Dirichlet([2.5, 3.5, 2.5]), [0.1, 0.6, 0.3])

    f = DualOptimalFiltering.create_dirichlet_mixture_marginals_pdf([1.5, 1.5, 1.5], [[0,0,0], [0,0,0], [0,0,0]], [1/3, 1/3, 1/3])

    for i in 1:3
        @test f(0.1)[i] ≈ (pdf(Beta(1.5, 3), 0.1) |> x -> repeat([x]; inner = 3) |> x -> x[i])
    end

    @test isreal(sum(DualOptimalFiltering.sample_from_Dirichlet_mixture([2.5, 3.5, 2.5], [[1,2,1], [1,2,1], [1,2,1]], [1/3, 1/3, 1/3])))

    @test mean(DualOptimalFiltering.sample_from_Dirichlet_mixture([2.5, 2.5, 2.5], [[1,2,1], [1,2,1], [1,2,1]], [1/3, 1/3, 1/3], 10000000), dims = 2) ≈ (([2.5, 2.5, 2.5] + [1,2,1]) |> x -> x/sum(x))  atol=10^(-3)

    @test DualOptimalFiltering.compute_quantile_mixture_beta([1.5, 1.5, 1.5], [[0,0,0], [0,0,0], [0,0,0]], [1/3, 1/3, 1/3], 0.5; marginal = 1) ≈ quantile(Beta(1.5, 3), 0.5)
    
end
