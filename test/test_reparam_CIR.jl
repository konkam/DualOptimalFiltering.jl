@testset "Testing the reparametrisation functions" begin
    δ, γ, σ = 1.1, 1.6, 2.3
    for pidx in eachindex([δ, γ, σ])
        @test [δ, γ, σ][pidx] ≈ DualOptimalFiltering.inverse_reparam_CIR(DualOptimalFiltering.reparam_CIR(δ, γ, σ)...)[pidx]
    end
    for pidx in eachindex([δ, γ, σ])
        @test [δ, γ, σ][pidx] ≈ DualOptimalFiltering.reparam_CIR(DualOptimalFiltering.inverse_reparam_CIR(δ, γ, σ)...)[pidx]
    end
end
