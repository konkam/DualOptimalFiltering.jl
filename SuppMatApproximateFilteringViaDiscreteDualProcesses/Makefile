################ CIR

saves_for_figures/filtering_error_varying_sigma.csv: scripts_for_figures/filtering_error_computation_CIR.jl scripts_for_figures/filtering_error_helper_functions_CIR.jl
	julia +1.8 --project=. scripts_for_figures/filtering_error_computation_CIR.jl

figures/convergence_predictive_distr_CIR.pdf: scripts_for_figures/convergence_predictive_dist_CIR.jl 
	julia +1.8 --project=. scripts_for_figures/convergence_predictive_dist_CIR.jl 

figures/filtering_error_comparison_CIR_2.pdf: scripts_for_figures/filtering_error_plot_CIR.R saves_for_figures/filtering_error_varying_sigma.csv 
	Rscript scripts_for_figures/filtering_error_plot_CIR.R

figures_CIR: figures/convergence_predictive_distr_CIR.pdf figures/filtering_error_comparison_CIR_2.pdf

################ Wright-Fisher

figures/convergence_predictive_distr_WF4.pdf: scripts_for_figures/convergence_predictive_dist_WF.jl scripts_for_figures/simulate_WF_data.jl 
	julia +1.8 --project=. scripts_for_figures/convergence_predictive_dist_WF.jl

figures/filtering_error_WF_2.pdf: scripts_for_figures/filtering_error_plot_WF.R saves_for_figures/filtering_error_WF.csv 
	Rscript scripts_for_figures/filtering_error_plot_WF.R

saves_for_figures/filtering_error_WF.csv: scripts_for_figures/filtering_error_computation_WF.jl scripts_for_figures/filtering_error_helper_functions_WF.jl scripts_for_figures/variability_of_prediction_WF.jl
	julia +1.8 --project=. --threads 1 scripts_for_figures/filtering_error_computation_WF.jl 

figures_WF: figures/convergence_predictive_distr_WF4.pdf figures/filtering_error_WF_2.pdf

################ Summary

all_figures: figures_CIR figures/convergence_predictive_distr_WF4.pdf figures_WF

# Dummy targets
.PHONY: all_figures figures_CIR figures_WF clean

# Clean target
clean:
	rm -f saves_for_figures/*.csv
	rm -f figures/*.pdf
