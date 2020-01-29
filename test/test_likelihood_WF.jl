using ExactWrightFisher

@testset "test the wf likelihood functions with array storage for precomputed coeffs" begin

    function simulate_WF3_data()
        K = 3
        α = ones(K)
        sα = sum(α)
        Pop_size_WF3 = 15
        Ntimes_WF3 = 3
        time_step_WF3 = 0.1
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        Random.seed!(4)
        wfchain_WF3 = Wright_Fisher_K_dim_exact_trajectory(rand(Dirichlet(K,0.3)), time_grid_WF3[1:(end-1)], α)
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α
    end
    data, wfchain_WF3, α = simulate_WF3_data()
    times = data |> keys |> collect |> sort


    current_logw = -0.5*ones(5,5,5)
    current_logw_prime = -0.5*ones(5,5,5)
    current_Λ_max = [2,1,3]
    current_Λ = DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max)
    y = [2,3,1]

    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef, smmax+1, smmax+1)

    @test_nowarn DualOptimalFiltering.precompute_next_terms_ar!(0, 12, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, 3.1, 0.2)

    @test_nowarn DualOptimalFiltering.WF_loglikelihood_from_adaptive_filtering(α, data, (x, y)-> (x, y); silence = false)

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_fixed_number(α, data, 3; silence = false)

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_fixed_fraction(α, data, 0.99; silence = false)

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_keep_above_threshold(α, data, 0.001; silence = false)


end
