using ExactWrightFisher, Random, Test, Distributions
using FeynmanKacParticleFilters

@testset "Testing the WF particle filtering functions" begin

    Random.seed!(0);
    α_vec = [1.2, 1.4, 1.3]
    K = length(α_vec)
    Pop_size_WF3 = 10
    Nparts = 100
    time_grid_WF3 = range(0, stop = 1, length = 10)
    wfchain = Wright_Fisher_K_dim_exact_trajectory([0.2, 0.4, 0.4], time_grid_WF3, α_vec)
    wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain[:,k])) for k in 1:size(wfchain,2)] |> l -> hcat(l...)
    data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))

    logGt = FeynmanKacParticleFilters.create_potential_functions(data_WF3, DualOptimalFiltering.multinomial_logpotential)
    Mt = DualOptimalFiltering.create_transition_kernels_WF(data_WF3, α_vec)
    RS(W) = rand(Categorical(W), length(W))

    # @test data_WF3[time_grid_WF3[2]] == [2, 4, 4]
    @test_nowarn Mt[time_grid_WF3[2]](α_vec / sum(α_vec))
    @test_nowarn pf_adaptive = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)

    data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t:t]' for t in 1:size(wfobs_WF3,2)]))

    @test_nowarn FeynmanKacParticleFilters.create_potential_functions(data_WF3, DualOptimalFiltering.multinomial_logpotential)

    @test_nowarn DualOptimalFiltering.fit_particle_filter_WF(data_WF3, α_vec; Nparts = 100)
end;
