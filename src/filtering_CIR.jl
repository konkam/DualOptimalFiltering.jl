function filter_CIR(δ, γ, σ, λ, data; silence = false)

    times = keys(data) |> collect |> sort
    Λ_of_t = Dict{Float64, Array{Int64,1}}()
    wms_of_t = Dict{Float64, Array{Float64,1}}()
    θ_of_t = Dict{Float64, Float64}()

    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params([1.], δ, γ/σ^2, λ, [0], data[0])

    Λ_of_t[0] = filtered_Λ
    wms_of_t[0] = filtered_wms # = 1.
    θ_of_t[0] = filtered_θ

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        filtered_θ, filtered_Λ, filtered_wms = get_next_filtering_distribution(filtered_Λ, filtered_wms, filtered_θ, times[k], times[k+1], δ, γ, σ, λ, data[times[k+1]])
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
        θ_of_t[times[k+1]] = filtered_θ
    end

    return Λ_of_t, wms_of_t, θ_of_t

end

function update_CIR_params(wms::Array{Ty,1}, δ::Real, θ::Real, λ::Real, Λ, y::Array{Tz,1}) where {Ty<:Number,Tz<:Integer}
    α = δ/2#Alternative parametrisation

    ny = sum(y)
    J = length(y)

    return θ + J*λ, Λ .+ ny, next_wms_from_wms_prime(wms, Λ, y, θ, α)
end

function get_next_filtering_distribution(current_Λ, current_wms, current_θ, current_time, next_time, δ, γ, σ, λ, next_y)
    predicted_θ, predicted_Λ, predicted_wms = predict_CIR_params(current_wms, δ, current_θ, γ, σ, current_Λ, next_time-current_time)
    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params(predicted_wms, δ, predicted_θ, λ, predicted_Λ, next_y)

    return filtered_θ, filtered_Λ, filtered_wms
end

function predict_CIR_params(wms::Array{Ty,1}, δ::Ty, θ::Ty, γ::Ty, σ::Ty, Λ, t::Ty) where Ty<:Number

    p = γ/σ^2*1/(θ*exp(2*γ*t) + γ/σ^2 - θ)
    θ_new = p * θ * exp(2*γ*t)
    Λ_new = 0:maximum(Λ)

    return θ_new, Λ_new, next_wms_prime_from_wms(wms, Λ, t, θ, γ, σ)

end
