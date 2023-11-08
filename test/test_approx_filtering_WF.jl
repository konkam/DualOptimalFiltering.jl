using ExactWrightFisher

@testset "test adaptive precomputation approx filtering functions" begin

    function simulate_WF3_data()
        K = 4
        α = ones(K)
        sα = sum(α)
        Pop_size_WF3 = 15
        Ntimes_WF3 = 3
        time_step_WF3 = 0.1
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        Random.seed!(4)
        wfchain_WF3 = Wright_Fisher_K_dim_exact_trajectory(rand(Dirichlet(K,0.3)), time_grid_WF3[1:(end-1)], α)
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α
    end
    data, wfchain_WF3, α = simulate_WF3_data()

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_number(α, data |> DualOptimalFiltering.prepare_WF_dat_1D_2D |> last, 100; silence = false)

    ## Check that Λ_of_t does not have duplicated entries
    for t in keys(Λ_of_t)
        @test length(unique(Λ_of_t[t])) == length(Λ_of_t[t])
    end

    Δt = 0.1


    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)
    DualOptimalFiltering.precompute_next_terms_ar!(0, smmax, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, Δt)

    sα = sum(α)
    tt = DualOptimalFiltering.WF_prediction_for_one_m_precomputed([10, 10, 10], sα, Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    for k in keys(tt)
        @test length(tt[k]) == length(unique(tt[k]))
    end


    Δt = 0.001


    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)
    DualOptimalFiltering.precompute_next_terms_ar!(0, smmax, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, Δt)

    sα = sum(α)
    tt = DualOptimalFiltering.WF_prediction_for_one_m_precomputed([10, 10, 10], sα, Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    for k in keys(tt)
        @test length(tt[k]) == length(unique(tt[k]))
    end


    @test_nowarn DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_number(α, data |> DualOptimalFiltering.prepare_WF_dat_1D_2D |> last, 5; silence = false)

    @test_nowarn DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_above_threshold(α, data |> DualOptimalFiltering.prepare_WF_dat_1D_2D |> last, 0.01; silence = false)

    @test_nowarn DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_fraction(α, data |> DualOptimalFiltering.prepare_WF_dat_1D_2D |> last, 0.95; silence = false)
end;
