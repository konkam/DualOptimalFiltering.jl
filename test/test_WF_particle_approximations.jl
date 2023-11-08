@testset "WF particle prediction tests" begin


    @test length(DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step_for_one_m([4,4,2], 0.1, 0.03)) == 3

    res = DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step([0.5, 0.3, 0.1, 0.1], 300., [[4,4,5], [2,3,2], [6,4,6], [4,4,1]], 0.3; nparts = 1000)

    @test res[1] == [[0,0,0]]
    @test res[2] == [1.]


    Random.seed!(0)

    function simulate_WF3_data()
        K = 4
        # α = ones(K)* 1. /K
        α = ones(K)* 3.
        sα = sum(α)
        Pop_size_WF3 = 15
        Ntimes_WF3 = 3
        time_step_WF3 = 0.1
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        Random.seed!(4)
        wfchain_WF3 = Wright_Fisher_K_dim_exact_trajectory(rand(Dirichlet(K,0.3)), time_grid_WF3[1:(end-1)], α)
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α, sα
    end
    data, wfchain_WF3, α, sα = simulate_WF3_data()

    dd = data |> DualOptimalFiltering.prepare_WF_dat_1D_2D |> last

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_number(α, dd, 5; silence = false)

    Λ = Λ_of_t[wms_of_t |> keys |> collect |> last] |> x -> x[1:3]

    ## Check that Λ_of_t does not have duplicated entries
    for t in keys(Λ_of_t)
        @test length(unique(Λ_of_t[t])) == length(Λ_of_t[t])
    end
    
    wms = wms_of_t[wms_of_t |> keys |> collect |> last]
    Λ = Λ_of_t[wms_of_t |> keys |> collect |> last]

    Δt = 0.1
    nparts = 1000


    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)
    DualOptimalFiltering.precompute_next_terms_ar!(0, smmax, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, Δt)


    res = DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step([1.], sα, Λ[1:1], Δt; nparts = nparts)
    ref = DualOptimalFiltering.predict_WF_params_precomputed([1.], sα, Λ[1:1], Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)
 
    R"library(tidyverse)"
    to_plot = R"tibble(lambda = $(ref[1] |> collect), wms = $(ref[2]), type = 'ref') %>% bind_rows(tibble(lambda = $(res[1]), wms = $(res[2]), type = 'res'))"
    R"$to_plot %>% mutate(lambda = as.character(lambda)) %>% arrange(lambda) %>% ggplot(aes(x=lambda, y = wms, colour = type)) + theme_bw() + geom_point() + geom_line()"


    res100 = DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step([1.], sα, Λ[1:1], Δt; nparts = 100)
    res1000 = DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step([1.], sα, Λ[1:1], Δt; nparts = 1000)
    res10000 = DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step([1.], sα, Λ[1:1], Δt; nparts = 10000)


    res_integrated100 = DualOptimalFiltering.WF_particle_integrated_PD_prediction_step([1.], sα, Λ[1:1], Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = 100)
    res_integrated1000 = DualOptimalFiltering.WF_particle_integrated_PD_prediction_step([1.], sα, Λ[1:1], Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = 1000)
    res_integrated10000 = DualOptimalFiltering.WF_particle_integrated_PD_prediction_step([1.], sα, Λ[1:1], Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = 10000)
    res_integrated100000 = DualOptimalFiltering.WF_particle_integrated_PD_prediction_step([1.], sα, Λ[1:1], Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = 100000)

    res_integrated10000_wrong = DualOptimalFiltering.WF_particle_integrated_PD_prediction_step_slower([1.], sα, Λ[1:1], Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = 10000)

    all_components = union(ref[1], 
    collect.(res[1]), collect.(res10000[1]), collect.(res_integrated100000[1])
    )

    gillespie_cv_comp = R"tibble(lambda = $all_components) %>%
    left_join(tibble(lambda = $(ref[1] |> x -> collect.(x)), Exact = $(ref[2]))) %>%
    mutate(Exact = ifelse(test = is.na(Exact), yes = 0, no = Exact)) %>%
    left_join(tibble(lambda = $(res100[1] |> x -> collect.(x)), PD_gillespie_100 = $(res100[2]))) %>%
    mutate(PD_gillespie_100 = ifelse(test = is.na(PD_gillespie_100), yes = 0, no = PD_gillespie_100)) %>%
    left_join(tibble(lambda = $(res1000[1] |> x -> collect.(x)), PD_gillespie_1000 = $(res1000[2]))) %>%
    mutate(PD_gillespie_1000 = ifelse(test = is.na(PD_gillespie_1000), yes = 0, no = PD_gillespie_1000))%>%
    left_join(tibble(lambda = $(res10000[1] |> x -> collect.(x)), PD_gillespie_10000 = $(res10000[2]))) %>%
    mutate(PD_gillespie_10000 = ifelse(test = is.na(PD_gillespie_10000), yes = 0, no = PD_gillespie_10000)) %>%
    left_join(tibble(lambda = $(res_integrated100[1] |> x -> collect.(x)), PD_integrated_100 = $(res_integrated100[2]))) %>%
    mutate(PD_integrated_100 = ifelse(test = is.na(PD_integrated_100), yes = 0, no = PD_integrated_100)) %>%
    left_join(tibble(lambda = $(res_integrated1000[1] |> x -> collect.(x)), PD_integrated_1000 = $(res_integrated1000[2]))) %>%
    mutate(PD_integrated_1000 = ifelse(test = is.na(PD_integrated_1000), yes = 0, no = PD_integrated_1000)) %>%
    left_join(tibble(lambda = $(res_integrated10000[1] |> x -> collect.(x)), PD_integrated_10000 = $(res_integrated10000[2]))) %>%
    mutate(PD_integrated_10000 = ifelse(test = is.na(PD_integrated_10000), yes = 0, no = PD_integrated_10000))  %>%
    left_join(tibble(lambda = $(res_integrated10000_slower[1] |> x -> collect.(x)), res_integrated10000_slower = $(res_integrated10000_slower[2]))) %>%
    mutate(res_integrated10000_slower = ifelse(test = is.na(res_integrated10000_slower), yes = 0, no = res_integrated10000_slower))"

    R"$gillespie_cv_comp %>%
        select(-lambda) %>%
        GGally::ggpairs()"

    res = DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step(wms, sα, Λ, Δt; nparts = nparts)
    ref = DualOptimalFiltering.predict_WF_params_precomputed(wms, sα, Λ, Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)
    R"library(tidyverse)"
    to_plot = R"tibble(lambda = $(ref[1] |> collect), wms = $(ref[2]), type = 'ref') %>% bind_rows(tibble(lambda = $(res[1]), wms = $(res[2]), type = 'res'))"
    R"$to_plot %>% mutate(lambda = as.character(lambda)) %>% ggplot(aes(x=lambda, y = wms, colour = type)) + theme_bw() + geom_point() + geom_line()"

    @test length(unique(Λ)) == length(Λ) #This showed that WF_particle_gillespie_PD_prediction_step_for_one_m had a side effect and changed m
    @test length(unique(res[1])) == length(res[1])
    @test length(unique(ref[1])) == length(ref[1])

    res2 = DualOptimalFiltering.WF_particle_integrated_PD_prediction_step(wms, sα, Λ, Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = nparts)


    res2 = DualOptimalFiltering.WF_neutral_particle_prediction_step_precomputed(wms, sα, Λ, Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = nparts)
    # This seems 5 times faster than the previous one

    res = DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step(wms, sα, Λ, Δt; nparts = nparts)


    res3 = DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_Moran_gillespie(wms, Λ, Δt, α; nparts = nparts)

    res4 = DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_WF_gillespie(wms, Λ, Δt, α; nparts = nparts)


    # WF_simulation_diffusion([0, 1, 20, 0], Δt, α)

    res5 = DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_WF_diffusion(wms, Λ, Δt, α; nparts = nparts)

    


    R"library(tidyverse)"
    all_components = union(ref[1], 
                            collect.(res[1]), 
                            res2[1] |> x -> collect.(x), 
                            res3[1] |> x -> collect.(x), 
                            res4[1] |> x -> collect.(x), 
                            res5[1] |> x -> collect.(x)
                            )

    to_plot = R"tibble(lambda = $all_components) %>%
                    left_join(tibble(lambda = $(ref[1] |> x -> collect.(x)), Exact = $(ref[2]))) %>%
                    mutate(Exact = ifelse(test = is.na(Exact), yes = 0, no = Exact)) %>% 
                    left_join(tibble(lambda = $(res[1] |> x -> collect.(x)), PD_gillespie = $(res[2]))) %>%
                    mutate(PD_gillespie = ifelse(test = is.na(PD_gillespie), yes = 0, no = PD_gillespie)) %>%
                    left_join(tibble(lambda = $(res2[1] |> x -> collect.(x)), PD_integrated = $(res2[2]))) %>%
                    mutate(PD_integrated = ifelse(test = is.na(PD_integrated), yes = 0, no = PD_integrated)) %>%
                    left_join(tibble(lambda = $(res3[1] |> x -> collect.(x)), BD_Dual_Moran_gillespie = $(res3[2]))) %>%
                    mutate(BD_Dual_Moran_gillespie = ifelse(test = is.na(BD_Dual_Moran_gillespie), yes = 0, no = BD_Dual_Moran_gillespie)) %>%
                    left_join(tibble(lambda = $(res4[1] |> x -> collect.(x)), BD_Dual_WF_gillespie = $(res4[2]))) %>%
                    mutate(BD_Dual_WF_gillespie = ifelse(test = is.na(BD_Dual_WF_gillespie), yes = 0, no = BD_Dual_WF_gillespie)) %>%
                    left_join(tibble(lambda = $(res5[1] |> x -> collect.(x)), BD_Dual_WF_diffusion = $(res5[2]))) %>%
                    mutate(BD_Dual_WF_diffusion = ifelse(test = is.na(BD_Dual_WF_diffusion), yes = 0, no = BD_Dual_WF_diffusion)) %>% 
                    mutate(lambda = as.character(lambda)) %>%
                    arrange(lambda) %>%
                    rowid_to_column(var = 'component_index')
                    "
    R"$to_plot %>% 
        gather(type, weight, -lambda, -component_index) %>%
        ggplot(aes(x=component_index, y = weight, colour = type)) + 
                 theme_bw() + 
                 geom_point() + 
                 geom_line()"

    R"$to_plot %>% 
        gather(type, weight, -lambda, -component_index) %>%
        ggplot(aes(x=component_index, y = weight, colour = type)) + 
                theme_bw() + 
                geom_point() + 
                geom_line() + 
                xlim(0, 20)"

    function compute_marginals(Λ, wms, nm; x_grid = range(0, 1, length = 30))
        res = DualOptimalFiltering.create_dirichlet_mixture_marginals_pdf(α, Λ, wms).(x_grid) |>
            v -> hcat(v...) |>
            transpose |> 
            collect |>
            m -> DataFrame(m, :auto) |>
            df -> @transform(df, :type = nm, :x = x_grid)
            return res
    end
    
    to_plot = compute_marginals(ref[1], ref[2], "Exact") |>
    df -> vcat(df, 
                compute_marginals(res[1], res[2], "PD_gillespie"),
                compute_marginals(collect.(res2[1]), res2[2], "PD_integrated"),
                compute_marginals(collect.(res3[1]), res3[2], "BD_Dual_Moran_gillespie"),
                compute_marginals(collect.(res4[1]), res4[2], "BD_Dual_WF_gillespie"),
                compute_marginals(collect.(res5[1]), res5[2], "BD_Dual_WF_diffusion")
    )

    R"$to_plot %>%
        tibble %>%
        gather(var, prob, x1:x4) %>%
        ggplot(aes(x = x, y = prob, colour = type)) + 
        theme_bw() + 
        facet_wrap(~var) + 
        geom_line()"
    # to_plot = R"tibble(lambda = $(ref[1] |> collect), wms = $(ref[2]), type = 'ref') %>% 
    #             bind_rows(tibble(lambda = $(res[1] |> x -> collect.(x)), wms = $(res[2]), type = 'PD_gillespie'),
    #             tibble(lambda = $(res2[1] |> x -> collect.(x)), wms = $(res2[2] |> collect), type = 'PD_integrated'),
    #             tibble(lambda = $(res3[1] |> x -> collect.(x)), wms = $(res3[2] |> collect), type = 'BD_Dual_Moran_gillespie'),
    #            6tibble(lambda = $(res4[1] |> x -> collect.(x)), wms = $(res4[2] |> collect), type = 'BD_Dual_WF_gillespie'),
    #             tibble(lambda = $(res5[1] |> x -> collect.(x)), wms = $(res5[2] |> collect), type = 'BD_Dual_WF_diffusion')
    #             ) %>%
    #             arrange(lambda) %>%
    #             rowid_to_column(var = 'component_index')"
    # R"$to_plot %>% 
    #     mutate(lambda = as.character(lambda)) %>% 
    #     ggplot(aes(x=component_index, y = wms, colour = type)) + 
    #         theme_bw() + 
    #         geom_point()"





end;

@testset "WF bootstrap prediction" begin


    Δt = 0.1
    res = DualOptimalFiltering.WF_particle_boostrap_prediction_step([1], [[1,1,1,1]], Δt, [1.,1.,1.,1.]; nparts = 10)

    @test size(res) == (4,10)
end;

@testset "WF dual filtering particle approximation" begin


    Random.seed!(0)

    function simulate_WF3_data()
        K = 4
        # α = ones(K)* 1. /K
        α = ones(K)* 3.
        sα = sum(α)
        Pop_size_WF3 = 15
        Ntimes_WF3 = 3
        time_step_WF3 = 0.1
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        Random.seed!(4)
        wfchain_WF3 = Wright_Fisher_K_dim_exact_trajectory(rand(Dirichlet(K,0.3)), time_grid_WF3[1:(end-1)], α)
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α, sα
    end
    data, wfchain_WF3, α, sα = simulate_WF3_data()

    dd = data |> DualOptimalFiltering.prepare_WF_dat_1D_2D |> last

    times = dd |> keys |> collect |> sort



    function WF_particle_PD_gillespie_prediction(wms, Λ, Δt, α, sα; nparts = 1000)
        DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step(wms, sα, Λ, Δt; nparts = nparts)
    end

    @test_nowarn DualOptimalFiltering.get_next_filtering_distribution_WF_particle_approx([[1,1,1,1]], [1.0], 0.1, 0.2, α, sα, dd[times |> last], WF_particle_PD_gillespie_prediction; nparts=10)

    @test_nowarn DualOptimalFiltering.filter_WF_particle_approx(α, dd, WF_particle_PD_gillespie_prediction; silence = false, trim0 = true, nparts=10)


    function WF_particle_BD_Moran_gillespie_prediction(wms, Λ, Δt, α, sα; nparts = 1000)
        DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_Moran_gillespie(wms, Λ, Δt, α; nparts = nparts)
    end


    @test_nowarn DualOptimalFiltering.filter_WF_particle_approx(α, dd, WF_particle_BD_Moran_gillespie_prediction; silence = false, trim0 = true, nparts=10)


    Δt = data |> keys |> collect |> sort |> diff |> unique |> first
    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)
    DualOptimalFiltering.precompute_next_terms_ar!(0, smmax, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, Δt)

    function WF_particle_PD_integrated_prediction(wms, Λ, Δt, α, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; nparts = 1000)
        DualOptimalFiltering.WF_particle_integrated_PD_prediction_step(wms, sα, Λ, Δt, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; nparts = nparts)
    end

    @test_nowarn DualOptimalFiltering.get_next_filtering_distribution_WF_particle_approx_precomputed([[1,1,1,1]], [1.0], 0.1, 0.2, α, sα, dd[times |> last], WF_particle_PD_integrated_prediction, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts=10)

    @test_nowarn DualOptimalFiltering.filter_WF_particle_approx_adaptive_precomputation_ar(α, dd, WF_particle_PD_integrated_prediction; silence = false, trim0 = true, nparts=10)

end;