function moran_simulation(starting_point, Δt, α)
    #K = length(starting_point)
    N = sum(starting_point)
    θ = sum(α)
    event_rate = 1/2*N*(θ+N)

    # initialisation
    t = 0.
    state = starting_point
    
    # simulation
    while t < Δt
        δt = rand(Exponential(1/event_rate))#The exponential distribution use the scale parametrisation
        i, j = choose_event(state, α, θ, N)
        state[i] -= 1
        state[j] +=1
        t += δt
        #println("t = $t, state = $state")
    end
    return state
end

function choose_event(state, α, θ, N)
    # an event can be coded as a pair i,j
    K = length(α)
    p = 0
    u = rand(Uniform(0,1))
    @inbounds for i in 1:K
        for j in 1:K
            p += state[i]*(α[j]+state[j])/(N*(θ+N))
            if p ≥ u
                return i,j
            end
        end
    end
    return 0, 0
end