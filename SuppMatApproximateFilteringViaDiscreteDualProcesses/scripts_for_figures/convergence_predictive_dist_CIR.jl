using RCall, DataFrames, DataFramesMeta, JLD, DualOptimalFiltering, Distributions, Random, StatsFuns


δ = 11.
npoints = 200
maxt = 20
x0 = 3 #Starting value
γ = 1.1
σ = 1.

mean_ = 5
nparts_grid = [50, 100, 500, 1000, 1500]

Random.seed!(1)
X = DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = maxt, length = npoints), x0, δ, γ, σ);
Y = map(λ -> rand(Poisson(λ), 1), X);
data = Dict(zip(range(0, stop = maxt, length = npoints), Y));

Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR(δ, γ, σ, 1.,data; silence=true, trim0=true);

forecast_time = 1. 
last_time = wms_of_t |> keys |> collect |> last
wms = wms_of_t[last_time]
Λ = Λ_of_t[last_time]
θ = θ_of_t[last_time]


ref = DualOptimalFiltering.predict_CIR_params(wms, 1., θ, γ, σ, Λ, forecast_time);

computed_approx_dual_predictions = Dict("particle_integrated_PD" => Dict(n_parts => DualOptimalFiltering.CIR_particle_integrated_PD_prediction_step(wms, Λ, forecast_time, θ, γ, σ; nparts=n_parts) for n_parts in nparts_grid),
                            "particle_BD" => Dict(n_parts => DualOptimalFiltering.CIR_particle_integrated_PD_prediction_step(wms, Λ, forecast_time, θ, γ, σ; nparts=n_parts) for n_parts in nparts_grid))

computed_PF_predictions = Dict(n_parts => DualOptimalFiltering.CIR_particle_PF_approx_prediction_step(wms, Λ, forecast_time, δ, θ, γ, σ; nparts=n_parts) for n_parts in nparts_grid)


grid = range(0, 15, length = 50)

compute_dens_dual = function(particle_dual, grid, δ, θ)
    particle_dual_pdf = DualOptimalFiltering.create_Gamma_mixture_pdf(δ, θ, particle_dual |> keys |> collect |> sort, getindex.(Ref(particle_dual), particle_dual |> keys |> collect |> sort))
    particle_dual_dens = particle_dual_pdf.(grid)
    return particle_dual_dens
end

compute_dens_pf = function(pf, grid)
    particle_PF_approx_pdf = DualOptimalFiltering.create_Gamma_mixture_density_smp(pf)
    particle_PF_approx_dens = particle_PF_approx_pdf.(grid)    
    return particle_PF_approx_dens
end

θ = ref[1]

ref_pdf = DualOptimalFiltering.create_Gamma_mixture_pdf(δ, ref[1], ref[2], ref[3])
ref_dens = ref_pdf.(grid)
methods = ["particle_integrated_PD", "particle_BD"]

to_plot = DataFrame(x = grid, method = "Ref", dens = ref_dens, n_particles = NaN) |>
    df -> vcat(df, [DataFrame(x = grid, method = met, dens = compute_dens_dual(computed_approx_dual_predictions[met][nparts], grid, δ, θ), n_particles = nparts) for met in methods, nparts in nparts_grid]...) |>
    df -> vcat(df, [DataFrame(x = grid, method = "PF", dens = compute_dens_pf(computed_PF_predictions[nparts], grid), n_particles = nparts) for nparts in nparts_grid]...)


R"library(tidyverse)"
R"""cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")"""

plt = R"""$to_plot %>%
            as_tibble %>%
            (function(dd){
                dd_ref = dd %>% filter(method == "Ref")
                dd_ref_repeated = lapply(X = $nparts_grid, FUN = function(nparts){dd_ref %>% mutate(n_particles = nparts)}) %>% bind_rows
                    dd %>%
                    filter(method != "Ref") %>%
                    bind_rows(dd_ref_repeated)
            }) %>%
            ggplot(aes(x = x, y = dens)) + 
            theme_bw() +
            facet_grid(~n_particles) + 
            geom_line(aes(colour = method)) +
            ylab("Predictive density") + 
            xlab("") + 
            # viridis::scale_colour_viridis(name = "", label = c("BD", "PD", "Bootstrap PF", "Exact"), discrete = T)
            scale_colour_manual(values=cbPalette, name = "", label = c("BD", "PD", "BPF", "Exact")) 
      """

R"
pdf('figures/convergence_predictive_distr_CIR.pdf', width = 8, height = 2)
plot($plt)
dev.off()
"
