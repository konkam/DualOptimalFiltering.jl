@testset "CIR smoothing tests" begin
    Random.seed!(1)

        δ = 3.
        γ = 2.5
        σ = 4.
        Nobs = 2
        dt = 0.011
        Nsteps = 10
        λ = 1.

        α = δ/2
        β = γ/σ^2

        time_grid = [k*dt for k in 0:(Nsteps-1)]
        X = DualOptimalFiltering.generate_CIR_trajectory(time_grid, 3, δ*1.2, γ/1.2, σ*0.7)
        Y = map(λ -> rand(Poisson(λ), Nobs), X);
        data = zip(time_grid, Y) |> Dict;

        @test_nowarn DualOptimalFiltering.CIR_smoothing_logscale_internals(δ, γ, σ, λ, data; silence = false)

end
