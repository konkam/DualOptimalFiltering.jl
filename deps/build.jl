using Pkg
println("I am being built...")
Pkg.add(PackageSpec(url="https://github.com/konkam/ExactWrightFisher.jl"))
Pkg.add(PackageSpec(url="https://github.com/konkam/FeynmanKacParticleFilters.jl"))
