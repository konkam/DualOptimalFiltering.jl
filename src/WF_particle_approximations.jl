function WF_particle_prediction_for_one_m_precomputed(m::Array{Int64,1}, sα::Ty, t::Ty, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff) where {Ty<:Number}
    # gm = map(x -> 0:x, m) |> vec |> x -> Iterators.product(x...)
    gm = indices_of_tree_below(m)

    U = rand(Uniform(0,1))

    function fun_n(n)
        i = m.-n
        # println(i)
        si = sum(i)
        sm = sum(m)
        return wm*(logpmmi_precomputed(i, m, sm, si, t, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff) |> exp)
    end

    w = 0

    for n in gm
        w += fun_n(n)
        if w > U
            return n
        end
    end

end

function WF_neutral_particle_prediction_step_precomputed(wms, Λ, Δt, α, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; nparts=1000)

    sα = sum(α)
    
    particle_sample = Λ[rand(Multinomial(nparts, Weights(wms)))]

    for i in eachindex(particle_sample)
        m = particle_sample[i]
        particle_sample[i] = WF_particle_prediction_for_one_m_precomputed(m, sα, Δt, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)
    end

    res = DataStructures.counter(particle_sample)

    ks = keys(res) |> collect

    return ks, [res[k]/nparts for k in ks]
    

end
