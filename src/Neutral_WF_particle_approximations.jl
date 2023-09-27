# function WF_particle_integrated_PD_prediction_step_wrong(wms::Array{Ty,1}, sα, Λ::Array{Array{Int64,1},1}, t, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = 1, nparts=1000) where {Ty<:Number}


#     # consider simplifying using the function particle_prediction_step_WF(wms, Λ, Δt, α, prediction_function; nparts=1000)


#     # Λ_wms_prime = Dict{Array{Int64,1}, Float64}()
#     # Λ_wms_prime = Dict{T, Float64}() where {T<:NTuple} # This is what comes out of indices_of_tree_below(m) This does not work infortunately.
#     Λ_wms_prime = Dict{NTuple{length(Λ[1]), Int64}, Float64}() # This is what comes out of indices_of_tree_below(m)

#     for i in 1:nparts
#         m = StatsBase.sample(Λ, Weights(wms))
#         # println("m = $m")
#         U = rand(Uniform())

#         gm = indices_of_tree_below(m)
#         w = 0.0
#         sm = sum(m)

#         for n in gm
#             i = m.-n
#             si = sum(i)
#             w += exp(logpmmi_precomputed(i, m, sm, si, t, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff))
#             # println("n = $n")
#             if w >= U
#                 if haskey(Λ_wms_prime, n)
#                     # Λ_wms_prime[n] += 1/nparts
#                     Λ_wms_prime[n] += 1
#                 else
#                     # Λ_wms_prime[n] = 1/nparts
#                     Λ_wms_prime[n] = 1
#                 end
#             end
#         end

#     end

#     Λ_prime = keys(Λ_wms_prime) |> collect
#     sw = Λ_wms_prime |> values |> sum

#     # println("sw = $sw")

#     return Λ_prime, Float64[Λ_wms_prime[m]/sw for m in Λ_prime]

# end

# function sample_idx(get_weight_i::Function)
#     U = rand(Uniform())
#     wsum::Float64 = get_weight_i(1)
#     i = 1
#     while wsum < U
#         println("wsum = $wsum, U = $U, i = $i")
#         i += 1
#         wsum += get_weight_i(i)
#     end
#     return i
# end

function WF_particle_integrated_PD_prediction_step(wms::Array{Ty,1}, sα, Λ::Array{Array{Int64,1},1}, t, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = 1, nparts=1000) where {Ty<:Number}

    # Λ_wms_prime = Dict{Array{Int64,1}, Float64}()
    # Λ_wms_prime = Dict{T, Float64}() where {T<:NTuple} # This is what comes out of indices_of_tree_below(m) This does not work infortunately.
    Λ_wms_prime = Dict{NTuple{length(Λ[1]), Int64}, Float64}() # This is what comes out of indices_of_tree_below(m)

    for i in 1:nparts
        m = StatsBase.sample(Λ, Weights(wms))
        # println("m = $m")

        gm = indices_of_tree_below(m)
        sm = sum(m)

        pmn = Array{Float64, 1}(undef, length(gm))

        idx = 1

        for n in gm
            i = m.-n
            si = sum(i)
            pmn[idx] = exp(logpmmi_precomputed(i, m, sm, si, t, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff))
            idx += 1
            # println("n = $n")
        end

        # println(sum(pmn))

        n = StatsBase.sample(collect(gm), Weights(pmn))

        if haskey(Λ_wms_prime, n)
            Λ_wms_prime[n] += 1/nparts
        else
            Λ_wms_prime[n] = 1/nparts
        end 

    end

    Λ_prime = keys(Λ_wms_prime) |> collect
    sw = Λ_wms_prime |> values |> sum

    return Λ_prime, Float64[Λ_wms_prime[m]/sw for m in Λ_prime]

end

# function WF_particle_gillespie_PD_prediction_step(wms, sα, Λ::Array{Array{Int64,1},1}, t; nparts = 1000) 

#     Λ_wms_prime = Dict{Array{Int64,1}, Float64}()

#     for i in 1:nparts
#         # println("At step $i Λ_wms_prime is $Λ_wms_prime")
#         # m = StatsBase.sample(Λ, Weights(wms))
#         m = Λ[rand(Categorical(Weights(wms)))]

#         if sum(m) == 0
#             n = m
#         else
#             n = WF_particle_gillespie_PD_prediction_step_for_one_m(m, sα, t)
#         end

#         # println("At step $i and n is $n")
#         # println("haskey(Λ_wms_prime, n) ? $(haskey(Λ_wms_prime, n))")


#         if haskey(Λ_wms_prime, n)
#             # println("already has this value ($n)")
#             # println("Before, was $Λ_wms_prime")
#             Λ_wms_prime[n] += 1/nparts
#             # println("Now, is $Λ_wms_prime")
#         else
#             Λ_wms_prime[n] = 1/nparts
#         end
#     end

#     Λ_prime = keys(Λ_wms_prime) |> collect
#     # println(Λ_wms_prime)
#     sw = Λ_wms_prime |> values |> sum
#     return Λ_prime, Float64[Λ_wms_prime[m]/sw for m in Λ_prime]
# end

# function WF_particle_gillespie_PD_prediction_step_for_one_m(m, sα, Δt) 

    
#     sm = sum(m)

#     if sm == 0
#         return m
#     else
#         state = deepcopy(m) # was mutating m

#         sumstate = sum(state)

#         event_rate = sumstate*(sα+sumstate-1)/2

#         # initialisation
#         t = rand(Exponential(1/event_rate)) #Double check
    
#         # simulation
#         while t < Δt && sum(state) > 0
#             # println(t)
#             # println(state ./ sum(state))
#             j = rand(Categorical(state ./ sum(state)))
#             state[j] -= 1
#             sumstate -= 1
#             event_rate = sumstate*(sα+sumstate-1)/2
#             δt = rand(Exponential(1/event_rate)) # The exponential distribution use the scale parametrisation
#             t += δt
#             #println("t = $t, state = $state")
#         end
#         return state
#     end

# end

# ## This is an alternative version to WF_particle_integrated_PD_prediction_step which seems 5 times faster
# function WF_neutral_particle_prediction_step_precomputed(wms, sα, Λ, Δt, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; nparts=1000)
  
#     # particle_sample = Λ[rand(Multinomial(nparts, Weights(wms)))]
#     particle_sample = StatsBase.sample(Λ, Weights(wms), nparts) |>
#                         x -> Tuple.(x)
#     # println(Λ)
#     # particle_sample = [StatsBase.sample(Λ, Weights(wms)) for i in 1:nparts]

#     for i in eachindex(particle_sample)
#         m = particle_sample[i]
#         particle_sample[i] = WF_particle_prediction_for_one_m_precomputed(m, sα, Δt, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)
#     end

#     res = DataStructures.counter(particle_sample)

#     ks = keys(res) |> collect

#     return ks, [res[k]/nparts for k in ks]
    

# end

# function WF_particle_prediction_for_one_m_precomputed(m, sα::Ty, t::Ty, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = 1) where {Ty<:Number}
#     # gm = map(x -> 0:x, m) |> vec |> x -> Iterators.product(x...)
#     gm = indices_of_tree_below(m)

#     U = rand(Uniform(0,1))

#     function fun_n(n)
#         i = m.-n
#         # println(i)
#         si = sum(i)
#         sm = sum(m)
#         return wm*(logpmmi_precomputed(i, m, sm, si, t, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff) |> exp)
#     end

#     w = 0

#     for n in gm
#         w += fun_n(n)
#         if w > U
#             return n
#         end
#     end

# end

# ##################### Birth-Death duality particle approximation


# function WF_neutral_particle_prediction_step_BD_Dual_Moran_gillespie(wms, Λ, Δt, α; nparts=1000)

#     particle_prediction_step_WF(wms, Λ, Δt, α, moran_simulation_Gillespie; nparts=nparts)
        
# end

# function WF_neutral_particle_prediction_step_BD_Dual_WF_gillespie(wms, Λ, Δt, α; nparts=1000)

#     particle_prediction_step_WF(wms, Λ, Δt, α, WF_discrete_Gillespie; nparts=nparts)
        
# end

# function WF_neutral_particle_prediction_step_BD_Dual_WF_diffusion(wms, Λ, Δt, α; nparts=1000)

#     particle_prediction_step_WF(wms, Λ, Δt, α, WF_simulation_diffusion; nparts=nparts)
    
# end


# function particle_prediction_step_WF(wms, Λ, Δt, α, prediction_function; nparts=1000)

    
#     particle_sample = StatsBase.sample(Λ, Weights(wms), nparts) #|>
#                         # x -> Tuple.(x)

#     for i in eachindex(particle_sample)
#         particle_sample[i] = prediction_function(particle_sample[i], Δt, α)
#     end

#     res = DataStructures.counter(particle_sample)

#     ks = keys(res) |> collect

#     return ks, [res[k]/nparts for k in ks]
    
# end

# function moran_simulation_Gillespie(starting_point, Δt, α)
#     #K = length(starting_point)
#     N = sum(starting_point)
#     if N == 0
#         return starting_point
#     else
#         θ = sum(α)
#         event_rate = 1/2*N*(θ+N)

#         # initialisation
#         t = rand(Exponential(1/event_rate)) #The exponential distribution use the scale parametrisation
#         state = deepcopy(starting_point)

#         while t < Δt
#             i, j = choose_event(state, α, θ, N)
#             state[i] -= 1
#             state[j] += 1
#             δt = rand(Exponential(1/event_rate)) #The exponential distribution use the scale parametrisation
#             t += δt
#             #println("t = $t, state = $state")
#         end
#         return state
#     end
# end

# function choose_event(state, α, θ, N)
#     # an event can be coded as a pair i,j
#     K = length(α)
#     p = 0
#     u = rand(Uniform(0,1))
#     @inbounds for i in 1:K
#         for j in 1:K
#             p += state[i]*(α[j]+state[j])/(N*(θ+N))
#             if p ≥ u
#                 return i,j
#             end
#         end
#     end
#     return 0, 0
# end


# function WF_discrete_Gillespie(starting_point, Δt, α)
#     N = sum(starting_point)
#     if N == 0
#         return starting_point
#     else
#         θ = sum(α)
#         k = length(α)

#         # Build mutation matrix for WF
#         Γ = repeat(α./(2*N), 1,k)' 
#         for i in 1:k
#             Γ[i,i] = 1 - sum(α./(2*N)) + α[i]/(2*N)
#         end

#         # Particle propagation
#         m = starting_point
#         for u in 1:floor(Int64, Δt * (N+θ))
#             m = vec(rand(Multinomial(N, Γ'*(m./N)),1))
#         end
#         return m
#     end
# end


# function from_simplex_to_M_space(x::Vector{Float64}, N)
#     x *= N
#     rest = 0.0
#     for (index, el) in enumerate(x)
#         x[index] = round(el + rest)
#         rest += (el - x[index])
#     end
#     return convert(Vector{Int64}, x)
# end


# function WF_simulation_diffusion(starting_point, Δt, α)
#     N = sum(starting_point)
#     if N == 0
#         return starting_point
#     else
#         Δt = convert(Float64, Δt)
#         return from_simplex_to_M_space(Wright_Fisher_K_dim_trajectory_with_t005_approx(starting_point./N, [Δt], α)[:, 2], N)
#     end
# end


# function random_simplex_point(k)
#     x = rand(k)
#     return x./sum(x)
# end


# ############### Bootstrap particle prediction

# function WF_particle_boostrap_prediction_step(wms, Λ::Array{Array{Int64,1},1}, t, α; nparts = 1000) 


#     particles = sample_from_Dirichlet_mixture(α, Λ, wms, nparts)

#     sα = sum(α)

#     for i in 1:nparts
#         particles[:,i] = Wright_Fisher_K_dim_transition_with_t005_approx(particles[:,i], t, α, sα)
#         # println(i)
#     end

#     return particles

# end



# ################ Particle approximate filtering functions

# function get_next_filtering_distribution_WF_particle_approx(current_Λ, current_wms, current_time, next_time, α, sα, next_y, predict_function; nparts=1000)

#     predicted_Λ, predicted_wms = predict_function(current_wms, current_Λ, next_time-current_time, α, sα; nparts = nparts)
    
#     filtered_Λ, filtered_wms = update_WF_params(predicted_wms, α, predicted_Λ, next_y)

#     return filtered_Λ, filtered_wms
# end


# function get_next_filtering_distribution_WF_particle_approx_precomputed(current_Λ, current_wms, current_time, next_time, α, sα, next_y, predict_function_precomputed, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; nparts=1000)
    
#     predicted_Λ, predicted_wms = predict_function_precomputed(current_wms, current_Λ, next_time-current_time, α, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; nparts = nparts)
    
#     filtered_Λ, filtered_wms = update_WF_params(predicted_wms, α, predicted_Λ, next_y)

#     return filtered_Λ, filtered_wms
# end



# function filter_WF_particle_approx(α, data, predict_function::Function; silence = false, trim0 = false, nparts=1000)

#     @assert length(α) == length(data[data |> keys |> first])

#     sα = sum(α)
#     times = keys(data) |> collect |> sort
#     Λ_of_t = Dict()
#     wms_of_t = Dict()

#     filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data[times[1]])
    
#     if trim0
#         mask = filtered_wms.>0
#         filtered_Λ, filtered_wms = filtered_Λ[mask], filtered_wms[mask]
#     end

#     Λ_of_t[times[1]] = filtered_Λ
#     wms_of_t[times[1]] = filtered_wms

#     for k in 1:(length(times)-1)
#         if (!silence)
#             println("Step index: $k")
#             println("Number of components: $(length(filtered_Λ))")
#         end

#         filtered_Λ, filtered_wms = get_next_filtering_distribution_WF_particle_approx(filtered_Λ, filtered_wms, times[k], times[k+1], α, sα, data[times[k+1]], predict_function; nparts=nparts)
#         if trim0 #Should be useless, components with weight 0 should disappear after sampling
#             mask = filtered_wms.>0
#             filtered_Λ, filtered_wms = filtered_Λ[mask], filtered_wms[mask]
#         end
#         Λ_of_t[times[k+1]] = filtered_Λ
#         wms_of_t[times[k+1]] = filtered_wms
#     end

#     return Λ_of_t, wms_of_t

# end

# function filter_WF_particle_approx_adaptive_precomputation_ar(α, data, predict_function_precomputed::Function; silence = false, return_precomputed_terms = false, trim0 = false, nparts=1000)
#     # println("filter_WF_mem2")

#     # @assert length(α) == length(data[collect(keys(data))[1]])
#     # println("$α, $(length(data[data |> keys |> first]))")
#     @assert length(α) == length(data[data |> keys |> first])
#     Δt = assert_constant_time_step_and_compute_it(data)

#     smmax = values(data) |> sum |> sum
#     log_ν_ar = Array{Float64}(undef, smmax, smmax)
#     log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
#     log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)

#     sα = sum(α)
#     times = keys(data) |> collect |> sort
#     Λ_of_t = Dict()
#     wms_of_t = Dict()

#     filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data[times[1]])

#     if trim0
#         mask = filtered_wms.>0
#         filtered_Λ, filtered_wms = filtered_Λ[mask], filtered_wms[mask]
#     end

#     Λ_of_t[times[1]] = filtered_Λ
#     wms_of_t[times[1]] = filtered_wms
#     new_sm_max = maximum(sum.(filtered_Λ))
#     precompute_next_terms_ar!(0, new_sm_max, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sα, Δt)
#     sm_max_so_far = new_sm_max

#     for k in 1:(length(times)-1)
#         if (!silence)
#             println("Step index: $k")
#             println("Number of components: $(length(filtered_Λ))")
#         end
#         last_sm_max = maximum(sum.(filtered_Λ))
#         new_sm_max = last_sm_max + sum(data[times[k+1]])

#         if sm_max_so_far < new_sm_max
#             precompute_next_terms_ar!(sm_max_so_far, new_sm_max, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sα, Δt)
#             sm_max_so_far = max(sm_max_so_far,new_sm_max)
#         end

#         filtered_Λ, filtered_wms = get_next_filtering_distribution_WF_particle_approx_precomputed(filtered_Λ, filtered_wms, times[k], times[k+1], α, sα, data[times[k+1]], predict_function_precomputed, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; nparts = nparts)
#         Λ_of_t[times[k+1]] = filtered_Λ
#         wms_of_t[times[k+1]] = filtered_wms
#     end

#     if return_precomputed_terms
#         return Λ_of_t, wms_of_t, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far
#     else
#         return Λ_of_t, wms_of_t
#     end

# end