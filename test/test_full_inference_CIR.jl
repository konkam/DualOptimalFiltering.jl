using ExactWrightFisher, Distributions

@testset "test full inference CIR" begin
    Random.seed!(0)
    times_sim = range(0, stop = 20, length = 20)
    X = DualOptimalFiltering.generate_CIR_trajectory(times_sim, 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),1), X);
    data = Dict(zip(times_sim, Y))
    δ, γ, σ, λ = 3., 0.5, 1., 1.
    Λ_of_t, logwms_of_t, θ_of_t, = DualOptimalFiltering.filter_CIR_logweights(δ, γ, σ, λ, data);

    times = Λ_of_t |> keys |> collect |> sort

    @test_nowarn DualOptimalFiltering.hidden_signal_loglikelihood_CIR(X, times, δ, γ, σ)

    @test_nowarn DualOptimalFiltering.emitted_data_conditional_loglikelihood_CIR(data, X, times, λ)

    @test_nowarn DualOptimalFiltering.joint_loglikelihood_CIR(data, X, times, δ, γ, σ, λ)


    prior_δ = truncated(Normal(5, 4), 0, Inf)
    prior_γ = truncated(Normal(5, 4), 0, Inf)
    prior_σ = truncated(Normal(5, 4), 0, Inf)
    prior_logpdf(δi, γi, σi) = logpdf(prior_δ, δi) + logpdf(prior_γ, γi) + logpdf(prior_σ, σi)

    @test_nowarn DualOptimalFiltering.joint_sampler_CIR_pruning_precompute(data, λ, prior_logpdf, [1.,1.,1.], 10, (x, y) -> (x, y), final_chain_length = 10)

    @test_nowarn DualOptimalFiltering.joint_sampler_CIR_keep_fixed_number_precompute(data, λ, prior_logpdf, [1.,1.,1.], 10, 2; final_chain_length = 10)

end;
