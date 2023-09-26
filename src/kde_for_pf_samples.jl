import StatsBase: RealVector, RealMatrix
using KernelEstimator, Optim

"""
    bwlscv(xdata::RealVector, kernel::Function)

Optimal bandwidth estimation using least square cross validation.
Adapted from https://github.com/panlanfeng/KernelEstimator.jl/blob/master/src/bandwidth.jl


# Examples
```julia-repl
julia> bar([1, 2], [1, 2])
1
```
"""
function bwlscv(xdata::RealVector, kernel::Function)
    n=length(xdata)
    w  = zeros(n)
    h0=bwnormal(xdata)
    #when n is large, leaveoneout is more expensive than numeric integration
    if kernel == gaussiankernel && n<200
        return Optim.minimizer(Optim.optimize(h -> KernelEstimator.Jh(xdata, h, w, n), 0.01*h0, 10*h0, iterations=200, abs_tol=h0/n))
    end

    xlb, xub = extrema(xdata)
    h0=KernelEstimator.midrange(xdata)
    if h0==0 #Algorithm returns 0 when h0=0. This happens when there are many ties, mixdrange can return 0
        h0 = median(xdata)
    end
    hlb = h0/n
    hub = h0
    if kernel == betakernel
        xlb = 0.0
        xub = 1.0
        hub = 0.25
        logxdata = log.(xdata)
        log1_xdata = log.(1.0 .- xdata)
        return Optim.minimizer(Optim.optimize(h -> KernelEstimator.Jh(xdata, logxdata, log1_xdata, kernel, h, w, n, xlb,xub), hlb, hub, iterations=200,abs_tol=h0/n^2))
    elseif kernel == gammakernel
        xlb = 0.0
        logxdata = log.(xdata)
        return Optim.minimizer(Optim.optimize(h -> KernelEstimator.Jh(xdata, logxdata, kernel, h, w, n, xlb,xub), hlb, hub, iterations=200,abs_tol=h0/n^2))
    end
    return Optim.minimizer(Optim.optimize(h -> KernelEstimator.Jh(xdata, kernel, h, w, n, xlb,xub), hlb, hub, iterations=200,abs_tol=h0/n^2))
end


"""
    create_gamma_kde_mixture_parameters(smp::Array{Float64,1})

This is using Chen, Song Xi. "Probability density function estimation using gamma kernels." Annals of the Institute of Statistical Mathematics 52, no. 3 (2000): 471-480.
Practically, given the bandwidth bw, the kernels are Gamma(α = x/bw+1, β = 1/bw) in the shape, rate parameterisation.
Returns the list of α for all kernels and the common β.


# Examples
```julia-repl
julia> bar([1, 2], [1, 2])
1
```
"""
function create_gamma_kde_mixture_parameters_one_β(smp::Array{T,1}) where T <: Real
    bw = bwlscv(smp, gammakernel)
    # if bw == 0
    #     bw = bwlcv(smp, gammakernel)
    # end
    # bw = bwlcv(smp, gammakernel)
    if bw==0
        error("bandwidth estimation by least square cross validation failed")
    else
        return smp ./ bw .+ 1, 1/bw
    end
end

function create_gamma_kde_mixture_parameters(smp::Array{T,1}) where T <: Real
    α_list, β = create_gamma_kde_mixture_parameters_one_β(smp)

    return α_list, repeat([β], length(α_list))
end


function create_Gamma_mixture_density_αβ(α_list::AbstractArray{T,1}, β_list::AbstractArray{U,1}, wms::AbstractArray{V,1}) where {T <: Real, U <: Real, V <: Real}
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    function res(x::Real)
        sum(wms[i] * pdf(Gamma(α_list[i], 1/β_list[i]), x) for i in eachindex(wms))
    end
    return res
end

function create_Gamma_mixture_density_smp(smp::Array{T,1}) where T <: Real
    α_list, β_list = create_gamma_kde_mixture_parameters(smp)

    w = 1. / length(α_list)
    wms = repeat([w], length(α_list))

    return create_Gamma_mixture_density_αβ(α_list, β_list, wms)
end

"""
create_Dirichlet_kde_mixture_parameters(xdata::RealMatrix)

This is using Aitchison, J., & Lauder, I. J. (1985). Kernel density estimation for compositional data. Applied Statistics, 129–137.
Optimal bandwidth selection is performed using a likelihood criterion, as in the article, which may not be the best choice.
The type RealMatrix is inherited from package StatsFuns

# Examples
```julia-repl
julia> using Distributions, Random
julia> Random.seed!(0);
julia> xdata = rand(Dirichlet([0.3,5.,2.3]), 5)
3×5 Array{Float64,2}:
 0.103154  0.000797253  0.0274887  0.0004612  0.069083
 0.525477  0.48484      0.620501   0.220089   0.500526
 0.371369  0.514362     0.352011   0.77945    0.430391
 julia> create_Dirichlet_kde_mixture_parameters(xdata)
 3×5 Array{Float64,2}:
 1.02705  1.00021  1.00721  1.00012  1.01812
 1.13781  1.12716  1.16273  1.05772  1.13127
 1.0974   1.1349   1.09232  1.20442  1.11288
```
"""
function create_Dirichlet_kde_mixture_parameters(xdata::RealMatrix)
    λ = DualOptimalFiltering.bwlcv(xdata, dirichletkernel)
    # if bw == 0
    #     bw = bwlcv(smp, gammakernel)
    # end
    # bw = bwlcv(smp, gammakernel)
    if λ==0
        error("bandwidth estimation by least square cross validation failed")
    else
        return 1 .+ xdata ./ λ
    end
end

function create_Dirichlet_kde_mixture_parameters(xdata::Array{Array{Float64,1},1})
    λ = DualOptimalFiltering.bwlcv(xdata, dirichletkernel)
    # if bw == 0
    #     bw = bwlcv(smp, gammakernel)
    # end
    # bw = bwlcv(smp, gammakernel)
    if λ==0
        error("bandwidth estimation by least square cross validation failed")
    else
        return Array{Float64,1}[1 .+ x ./ λ for x in xdata]
    end
end
