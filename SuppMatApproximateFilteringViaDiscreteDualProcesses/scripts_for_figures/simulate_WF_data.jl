using DualOptimalFiltering, Random, Distributions, ExactWrightFisher

function simulate_WF_data(;K = 4, α_sym = 3)
    
    α = ones(K)* α_sym
    sα = sum(α)
    Pop_size_WF = 15
    Ntimes_WF = 3
    time_step_WF = 0.1
    time_grid_WF = [k*time_step_WF for k in 0:(Ntimes_WF-1)]
    Random.seed!(4)
    wfchain_WF = Wright_Fisher_K_dim_exact_trajectory(rand(Dirichlet(K,0.3)), time_grid_WF[1:(end-1)], α)
    wfobs_WF = [rand(Multinomial(Pop_size_WF, wfchain_WF[:,k])) for k in 1:size(wfchain_WF,2)] |> l -> hcat(l...)
    data_WF = Dict(zip(time_grid_WF , [wfobs_WF[:,t] for t in 1:size(wfobs_WF,2)]))
    return data_WF, wfchain_WF, α, sα
end


