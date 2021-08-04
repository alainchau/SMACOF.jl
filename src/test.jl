using SMACOF
using Test
using Distances
using LinearAlgebra
using Random
using Statistics
Random.seed!(1993)

# X = rand(2, 200)
# Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
# Y = wda_smacof(dists(Y), η=0.95, verbose=false, anchors=X, ε=1e-12)
# println(norm(X - Y))

function main()
    X = randn(2, 150)
    Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
    @time Y = fit(Smacof(X, verbose=false, itmax=100), anchors=X)
    println()
end