using SpecialFunctions, StatsFuns, Base.Iterators


"""
    log_int_prod_2_Gammas(α1, β1, α2, β2)

Compute the log of ``\\int_{\\mathbb{R}^+} Gamma(α1, β1)Gammas(α2, β2)`` exactly. This formula is only valid for ``α1 + α2 > 1``.

# Examples
```julia-repl
julia> log_int_prod_2_Gammas(1, 1, 1, 1) #== -log(2)
-0.6931471805599453
julia> log_int_prod_2_Gammas(2, 0.5, 1, 0.5) #==  log(0.5^2*0.5 )
-0.6931471805599453
```
"""
function log_int_prod_2_Gammas(α1, β1, α2, β2)
    @assert α1 + α2 > 1 "The formula for the L2 distance is only valid for α1 + α2 > 1"
    return α1 * log(β1) + α2 * log(β2) + lgamma_local(α1 + α2 - 1) - lgamma_local(α1) - lgamma_local(α2) - (α1 + α2 - 1) * log(β1 + β2)
end

"""
    int_prod_2_Gammas(α1, β1, α2, β2)

Compute ``\\int_{\\mathbb{R}^+} Gamma(α1, β1)Gammas(α2, β2)`` exactly. This formula is only valid for ``α1 + α2 > 1``.

# Examples
```julia-repl
julia> int_prod_2_Gammas(1, 1, 1, 1)
0.5
julia> int_prod_2_Gammas(2, 0.5, 1, 0.5) #==  0.5^2*0.5
-0.6931471805599453
```
"""
int_prod_2_Gammas(α1, β1, α2, β2) = log_int_prod_2_Gammas(α1, β1, α2, β2) |> exp


function log_L2_dist_Gamma_mixtures(logw1, αlist_1, βlist_1, logw2, αlist_2, βlist_2)
    I = length(logw1)
    J = length(logw2)
    first_logterm = logsumexp(logw1[i] + logw1[j] + log_int_prod_2_Gammas(αlist_1[i], βlist_1[i], αlist_1[j], βlist_1[j]) for (i,j) in Base.Iterators.product(1:I, 1:I))

    second_logterm =  logsumexp(logw2[i] + logw2[j] + log_int_prod_2_Gammas(αlist_2[i], βlist_2[i], αlist_2[j], βlist_2[j]) for (i,j) in Base.Iterators.product(1:J, 1:J))

    third_log_term = logsumexp(log(2) + logw1[i] + logw2[j] + log_int_prod_2_Gammas(αlist_1[i], βlist_1[i], αlist_2[j], βlist_2[j]) for (i,j) in Base.Iterators.product(1:I, 1:J))

    # println([first_logterm, second_logterm, third_log_term])
    # println(exp.([first_logterm, second_logterm, third_log_term]))
    # println(sum(exp.([first_logterm, second_logterm, third_log_term]) .* [1,1,-1]))
    #
    # lx = [first_logterm, second_logterm, third_log_term]
    # signs = [1,1,-1]
    # m = maximum(lx)
    #
    # println("m= $m")
    # scaled_sum = sum(signs .* exp.(lx .- m))
    # println("scaled_sum= $scaled_sum")
    #
    # println("abs(scaled_sum) <= eps(Float64): $(abs(scaled_sum) <= eps(Float64))")

    log_squared_L2_dist = ExactWrightFisher.signed_logsumexp([first_logterm, second_logterm, third_log_term], [1,1,-1])[2] #The result will always be positive
    return 0.5*log_squared_L2_dist
end

"""
    log_int_prod_2_Dir(α1, α2)

Compute the log of ``\\int_{\\nabla_K} Dir(α1)Dir(α2)`` exactly. This formula is only valid for ``\\forall j \\in \\{1,\\ldots, K\\}, \\alpha_{1,j} + \\alpha_{2,j} > 1``.

# Examples
```julia-repl
julia> DualOptimalFiltering.log_int_prod_2_Dir([1,1], [1,1])
0.0
julia> log_int_prod_2_Gammas(2, 0.5, 1, 0.5) #==  log(0.5^2*0.5 )
-0.6931471805599453
```
"""
function log_int_prod_2_Dir(α1, α2)
    @assert all(α1 .+ α2 .> 1) "The formula for the L2 distance is only valid for α1 + α2 > 1 for each component of α"
    return lmvbeta(α1 .+ α2 .- 1) - lmvbeta(α1) - lmvbeta(α2)
end


"""
    lmvbeta(α::AbstractArray{T,1}) for T <: Real

Compute the log of the [multivariate beta function](https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function): ``\\Beta(\\alpha_1,\\alpha_2,\\ldots\\alpha_n) = \\frac{\\Gamma(\\alpha_1)\\Gamma(\\alpha_2) \\cdots \\Gamma(\\alpha_n)}{\\Gamma(\\alpha_1 + \\alpha_2 + \\cdots + \\alpha_n)}``.

# Examples
```julia-repl
julia> lmvbeta([1,1])
0.0
```
"""
function lmvbeta(α::AbstractArray{T,1}) where T <: Real
    return sum(lgamma_local.(α)) - lgamma_local(sum(α))
end


# function log_L2_dist_Dirichlet_mixtures(logw1::RealVector, αlist_1::Array{RealVector, 1}, logw2::RealVector, αlist_2::Array{RealVector, 1})
function log_L2_dist_Dirichlet_mixtures(logw1::RealVector, αlist_1::Array{T, 1}, logw2::RealVector, αlist_2::Array{T, 1}) where T <: RealVector
    I = length(logw1)
    J = length(logw2)
    first_logterm = logsumexp(logw1[i] + logw1[j] + log_int_prod_2_Dir(αlist_1[i], αlist_1[j]) for (i,j) in Base.Iterators.product(1:I, 1:I))
    second_logterm =  logsumexp(logw2[i] + logw2[j] + log_int_prod_2_Dir(αlist_2[i], αlist_2[j]) for (i,j) in Base.Iterators.product(1:J, 1:J))
    third_log_term = logsumexp(log(2) + logw1[i] + logw2[j] + log_int_prod_2_Dir(αlist_1[i], αlist_2[j]) for (i,j) in Base.Iterators.product(1:I, 1:J))

    # println([first_logterm, second_logterm, third_log_term])
    # println(exp.([first_logterm, second_logterm, third_log_term]))
    # println(sum(exp.([first_logterm, second_logterm, third_log_term]) .* [1,1,-1]))
    #
    # lx = [first_logterm, second_logterm, third_log_term]
    # signs = [1,1,-1]
    # m = maximum(lx)
    #
    # println("m= $m")
    # scaled_sum = sum(signs .* exp.(lx .- m))
    # println("scaled_sum= $scaled_sum")
    #
    # println("abs(scaled_sum) <= eps(Float64): $(abs(scaled_sum) <= eps(Float64))")

    log_squared_L2_dist = ExactWrightFisher.signed_logsumexp([first_logterm, second_logterm, third_log_term], [1,1,-1])[2] #The result will always be positive
    return 0.5*log_squared_L2_dist
end

# DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.6,0.4]), [[1,2,1],[4,2,3]])
# #
# DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]])
#
#
# DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.200001,0.3,0.5] |> DualOptimalFiltering.normalise), [[1,2,3],[1,2,4],[5,3,2]])
#
# DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.2000001,0.3,0.5] |> DualOptimalFiltering.normalise), [[1,2,3],[1,2,4],[5,3,2]])
#
# DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.20000001,0.3,0.5] |> DualOptimalFiltering.normalise), [[1,2,3],[1,2,4],[5,3,2]])
#
# DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.2000000001,0.3,0.5] |> DualOptimalFiltering.normalise), [[1,2,3],[1,2,4],[5,3,2]])
#
# import DualOptimalFiltering.RR
#
# ExactWrightFisher.signed_logsumexp([log(RR(3.00000000000001)), log(RR(3)), log(RR(6))],[1,1,-1])
#
# ExactWrightFisher.signed_logsumexp([log(3), log(3), log(6)],[1,1,-1])
