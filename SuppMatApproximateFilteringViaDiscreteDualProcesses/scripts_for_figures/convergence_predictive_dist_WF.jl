using RCall, DataFramesMeta, Random, DualOptimalFiltering, KernelEstimator

R"library(tidyverse)"

include("simulate_WF_data.jl")


function compute_marginals(Λ, wms, nm; x_grid = range(0, 1, length = 50))
    res = DualOptimalFiltering.create_dirichlet_mixture_marginals_pdf(α, Λ, wms).(x_grid) |>
        v -> hcat(v...) |>
        transpose |> 
        collect |>
        m -> DataFrame(m, :auto) |>
        df -> @transform(df, :type = nm, :x = x_grid)
        return res
end

function compute_marginals_particles(xdata, nm; x_grid = range(0, 1, length = 50))

    #Caution ! The particles in xdata are transposed inside the function to match the functions in DualOptimalFiltering dirichlet_kde.jl

    xdata_t = xdata'

    bw = DualOptimalFiltering.bwlcv_large_bounds(xdata, dirichletkernel)


    res = [DualOptimalFiltering.dirichletkernel_marginals.(x_grid, i, Ref(xdata_t), bw; log = false) for i in 1:size(xdata_t, 2)] |>
        v -> hcat(v...) |>
        m -> DataFrame(m, :auto) |>
        df -> @transform(df, :type = nm, :x = x_grid)

    return res
end

function compute_marginals_particles2(xdata, nm; x_grid = range(0, 1, length = 50))

    res = [KernelEstimator.kerneldensity(xdata[i,:], xeval = x_grid, kernel = betakernel) for i in 1:size(xdata, 1)] |>
        v -> hcat(v...) |>
        m -> DataFrame(m, :auto) |>
        df -> @transform(df, :type = nm, :x = x_grid)

    return res
end

Random.seed!(0)
K = 4

data, wfchain_WF, α, sα = simulate_WF_data(K=K, α_sym = 3.)

dd = data |> DualOptimalFiltering.prepare_WF_dat_1D_2D |> last

Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_number(α, dd, 5; silence = false)

times = Λ_of_t |> keys |> collect |> sort

Λ = Λ_of_t[times |> last]
wms = wms_of_t[times |> last]

Δt = times[end] - times[end-1]

smmax = sum.(Λ) |> maximum
log_ν_ar = Array{Float64}(undef, smmax, smmax)
log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)
DualOptimalFiltering.precompute_next_terms_ar!(0, smmax, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, Δt)

ref = DualOptimalFiltering.predict_WF_params_precomputed(wms, sα, Λ, Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

nparts_grid = [100, 500, 1000, 5000]


computed_approx_dual_predictions = Dict(
    #"PD Gillespie" => Dict(n_parts => DualOptimalFiltering.WF_particle_gillespie_PD_prediction_step(wms, sα, Λ, Δt; nparts=n_parts) for n_parts in nparts_grid),
                                        "PD" => Dict(n_parts => DualOptimalFiltering.WF_particle_integrated_PD_prediction_step(wms, sα, Λ, Δt, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts=n_parts) for n_parts in nparts_grid),
                                        "BD Gillespie Moran" => Dict(n_parts => DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_Moran_gillespie(wms, Λ, Δt, α; nparts=n_parts) for n_parts in nparts_grid),
                                        "BD Gillespie WF" => Dict(n_parts => DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_WF_gillespie(wms, Λ, Δt, α; nparts=n_parts) for n_parts in nparts_grid),
                                        "BD diffusion WF" => Dict(n_parts => DualOptimalFiltering.WF_neutral_particle_prediction_step_BD_Dual_WF_diffusion(wms, Λ, Δt, α; nparts=n_parts) for n_parts in nparts_grid))


computed_PF_predictions = Dict(n_parts => DualOptimalFiltering.WF_particle_boostrap_prediction_step(wms, Λ, Δt, α; nparts=n_parts) for n_parts in nparts_grid)

function second(x) 
    return x[2] 
end

to_plot = compute_marginals(ref[1], ref[2], "Exact") |>
            df -> @transform(df, :n_particles = NaN) |>
            df -> vcat(df, [compute_marginals(computed_approx_dual_predictions[method][nparts] |> first, 
                                                computed_approx_dual_predictions[method][nparts] |> second, method) |> df -> @transform(df, :n_particles = nparts) for nparts in nparts_grid, method in keys(computed_approx_dual_predictions)]...) |>
            df -> vcat(df, [compute_marginals_particles2(computed_PF_predictions[nparts], "bootstrap_WF") |> df -> @transform(df, :n_particles=nparts) for nparts in nparts_grid]...)

R"""methods_order = c("Exact", 
"PD", 
#"PD Gillespie", 
"BD Gillespie Moran", "BD Gillespie WF", "BD diffusion WF", "Bootstrap PF")"""

R"""cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")"""


plt = R"""$to_plot %>%
        as_tibble %>%
        (function(dd){
            dd_ref = dd %>% filter(type == "Exact")
            dd_ref_repeated = lapply(X = $nparts_grid, FUN = function(nparts){dd_ref %>% mutate(n_particles = nparts)}) %>% bind_rows
                dd %>%
                filter(type != "Exact") %>%
                bind_rows(dd_ref_repeated)
        }) %>%
        filter(type != "PD_Gillespie") %>%
        mutate(type = gsub("bootstrap_WF", "Bootstrap PF", type)) %>%
        mutate(type = factor(type, levels = methods_order)) %>%
        ggplot(aes(x = x, y = x1)) + 
        theme_bw() +
        facet_grid(~n_particles) + 
        geom_line(aes(colour = type)) +
        ylab("Predictive density (1st dim)") + 
        xlab(expression(x[1])) + 
        # viridis::scale_colour_viridis(name = "Method", 
        #                                 # label = c("Birth-Death", "Pure Death", "Bootstrap PF", "Exact"),
        #                                 discrete = T, direction = -1)
        scale_colour_manual(name = "", values=cbPalette)
      """            

R"
pdf('figures/convergence_predictive_distr_WF4.pdf', width = 11, height = 2)
plot($plt)
dev.off()
"