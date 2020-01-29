import StatsBase: RealVector, RealMatrix
using KernelEstimator, Optim
import KernelEstimator.lcv, KernelEstimator.bwlcv, KernelEstimator.midrange

function dirichletkernel_oneval(x::RealVector, xdata::RealVector, λ::Real; log = false)
    if(log)
        logpdf(Dirichlet(1 .+ xdata/λ), x)
    else
        pdf(Dirichlet(1 .+ xdata/λ), x)
    end
end

function dirichletkernel(x::RealVector, xdata::RealMatrix, λ::Real, w::Vector, n::Int; log = false)
    # w .= mapslices(xdatavec -> dirichletkernel_oneval(x, xdatavec, λ), xdata, 2) |> vec
    for i in eachindex(w)
        w[i] = dirichletkernel_oneval(x, xdata[i,:], λ; log = log)
    end
    nothing
end

function dirichletkernel(x::RealVector, xdata::Array{Array{Float64,1},1}, λ::Real, w::Vector, n::Int; log = false)
    # w .= mapslices(xdatavec -> dirichletkernel_oneval(x, xdatavec, λ), xdata, 2) |> vec
    for i in eachindex(w)
        w[i] = dirichletkernel_oneval(x, xdata[i], λ; log = log)
    end
    nothing
end

function midrange(x::RealMatrix)
    mapslices(xvec -> quantile(xvec, [.25, .75]), x, dims = 1) |> x -> x[2,:] - x[1,:] |> maximum
#     lq, uq = quantile(x, [.25, .75])
#     uq - lq
end

# midrange(x::Array{Array{Float64,1},1}) = midrange.(x) |> maximum
# midrange(x::Array{Array{Int64,1},1}) = midrange.(x) |> maximum

ff(x)::Float64 = max(midrange([xi[1] for xi in x]), midrange([xi[2] for xi in x]), midrange([xi[3] for xi in x]))

midrange(x::Array{Array{Float64,1},1})::Float64 = ff(x)

midrange(x::Array{Array{Int64,1},1})::Float64 = ff(x)

function lcv(xdata::RealMatrix, kernel::Function, h::Real, w::Vector, n::Int)
#     -mean(kerneldensity(xdata,xdata,kernel,h)) + mean(map(kernel, xdata, xdata, h))
    ind = 1
    ind_end = 1+n
    ll = 0.0
    @inbounds while ind < ind_end
        # kernel(xdata[ind,:], xdata, h, w, n)
        kernel(view(xdata, ind, :), xdata, h, w, n)
        w[ind] = 0.0
        ll += log(mean(w))
        ind += 1
    end
    -ll
end

function lcv(xdata::Array{Array{Float64,1},1}, kernel::Function, h::Real, w::Vector, n::Int)
    #     -mean(kerneldensity(xdata,xdata,kernel,h)) + mean(map(kernel, xdata, xdata, h))
    ind = 1
    ind_end = 1+n
    ll = 0.0
    @inbounds while ind < ind_end
        kernel(xdata[ind], xdata, h, w, n)
        w[ind] = 0.0
        ll += log(mean(w))
        ind += 1
    end
    -ll
end

function bwlcv(xdata::RealMatrix, kernel::Function)
    #This function seems to hit against the higher bound
    n = size(xdata,1)
    w = zeros(n)
    h0=midrange(xdata)
    if h0==0 #Algorithm returns 0 when h0=0. This happens when there are many ties, mixdrange can return 0
        h0 = median(xdata)
    end
    hlb = h0/n^2
    hub = h0
    if kernel==betakernel
        hub = 0.25
    end
    return Optim.minimizer(Optim.optimize(h->lcv(xdata,kernel,h,w,n), hlb, hub, iterations=200,abs_tol=h0/n^2))
end

function bwlcv(xdata::Array{Array{Float64,1},1}, kernel::Function)
    n = length(xdata)
    w = zeros(n)
    h0=midrange(xdata)
    if h0==0 #Algorithm returns 0 when h0=0. This happens when there are many ties, mixdrange can return 0
        h0 = median(xdata |> Base.Iterators.flatten)
    end
    hlb = h0/n^2
    hub = h0
    if kernel==betakernel
        hub = 0.25
    end
    return Optim.minimizer(Optim.optimize(h->lcv(xdata,kernel,h,w,n), hlb, hub, iterations=200,abs_tol=h0/n^2))
end

function bwlcv_large_bounds(xdata::RealMatrix, kernel::Function)
    n = size(xdata,1)
    w = zeros(n)
    h0=midrange(xdata)
    if h0==0 #Algorithm returns 0 when h0=0. This happens when there are many ties, mixdrange can return 0
        h0 = median(xdata)
    end
    hlb = h0/n^2
    hub = h0*10
    if kernel==betakernel
        hub = 0.25
    end
    res = Optim.minimizer(Optim.optimize(h->lcv(xdata,kernel,h,w,n), hlb, hub, iterations=200,abs_tol=h0/n^2))
    i = 1
    while (res==hub)&(i<10)
        res = Optim.minimizer(Optim.optimize(h->lcv(xdata,kernel,h,w,n), hlb, 10*i*hub, iterations=200,abs_tol=h0/n^2))
        i = i+1
    end
    return res
end

function minus_log_leaveoneout(xdata::RealMatrix, kernel::Function, h::Real, w::Vector, n::Int)

    ll = 0.0
    @inbounds for ind in 1:n
        kernel(view(xdata, ind, :), xdata, h, w, n; log = false)
        w[ind] = 0.0
        ll += log(sum(w))
    end
    - ll + log(n-1)
end

function bwloo_large_bounds(xdata::RealMatrix, kernel::Function)
    n = size(xdata,1)
    w = zeros(n)
    h0=midrange(xdata)
    if h0==0 #Algorithm returns 0 when h0=0. This happens when there are many ties, mixdrange can return 0
        h0 = median(xdata)
    end
    hlb = h0/n^2
    hub = h0#*10
    if kernel==betakernel
        hub = 0.25
    end
#     println(hlb)
#     println(hub)
#     res = Optim.minimizer(Optim.optimize(h->lcv(xdata,kernel,h,w,n), hlb, hub, iterations=200,abs_tol=h0/n^2))
    res = Optim.minimizer(Optim.optimize(h->minus_log_leaveoneout(xdata,kernel,h,w,n), hlb, hub, iterations=200))
    i = 1
    while (isapprox(res,hub, atol = h0/n^2))&(i<10)
        res = Optim.minimizer(Optim.optimize(h->minus_log_leaveoneout(xdata,kernel,h,w,n), hlb, 10^i*hub, iterations=200,abs_tol=h0/n^2))
        i = i+1
    end
    return res
end
