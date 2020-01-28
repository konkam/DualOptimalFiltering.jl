function update_logweights_cost_to_go(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_prime_kp2, ykp1, params::NamedTuple, logμπh::Function)
    return Float64[logwms_tilde_kp2[k]  + logμπh(Λ_tilde_prime_kp2[k], θ_tilde_prime_kp2, ykp1, params) for k in eachindex(Λ_tilde_prime_kp2)]
end
