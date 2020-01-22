using Random, Distributions

@testset "test MCMC sampler on CIR" begin
    function simulate_CIR_data(;Nsteps_CIR = 200)
        Random.seed!(1)

        δ = 3.
        γ = 2.5
        σ = 2.
        Nobs = 5
        dt_CIR = 0.011
        # max_time = .1
        # max_time = 0.001
        λ = 1.

        time_grid_CIR = [k*dt_CIR for k in 0:(Nsteps_CIR-1)]
        X_CIR = DualOptimalFiltering.generate_CIR_trajectory(time_grid_CIR, 3, δ, γ, σ)
        Y_CIR = map(λ -> rand(Poisson(λ), Nobs), X_CIR);
        data_CIR = Dict(zip(time_grid_CIR, Y_CIR))
        return data_CIR, X_CIR,  δ, γ, σ, λ
    end
    data_CIR, X_CIR,  δ, γ, σ, λ = simulate_CIR_data(;Nsteps_CIR = 40)
    prior_δ = truncated(Normal(5, 4), 0, Inf)
    prior_γ = truncated(Normal(5, 4), 0, Inf)
    prior_σ = truncated(Normal(5, 4), 0, Inf)
    prior_logpdf(δi, γi, σi) = logpdf(prior_δ, δi) + logpdf(prior_γ, γi) + logpdf(prior_σ, σi)

    function unnormalised_logposterior(δ, γ, σ)::Float64
    prior_contribution = prior_logpdf(δ, γ, σ)
        if isinf(prior_contribution)
            return prior_contribution
        else
            return prior_contribution + full_loglikelihood(δ, γ, σ, λ, data_CIR)::Float64
        end
    end

    full_loglikelihood(δ, γ, σ, λ, data, fun::Function)::Float64 = rand(Normal())
    full_loglikelihood(δ, γ, σ, λ, data)::Float64 = full_loglikelihood(δ, γ, σ, λ, data, x -> x)


    unnormalised_logposterior_vec(v) = unnormalised_logposterior(v...)

    @test_nowarn DualOptimalFiltering.get_mcmc_samples(15, [1.,1.,1.], DualOptimalFiltering.Jtnorm_create, unnormalised_logposterior_vec)

end;
