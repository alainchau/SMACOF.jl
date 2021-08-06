using SMACOF
using Test
using Distances
using LinearAlgebra
using Random
using Statistics
using BenchmarkTools
# using Plots
# ENV["GKSwstype"] = "100"
Random.seed!(1993)

# @testset "stress/distortion" begin
#     A = [0.0    0; 0    1; 1   0]
#     B = [0.0   .1 ; 0  1; 1   0]
#     n = size(A, 1)
#     s = (0.9 - 1)^2 + (sqrt(1 + 0.1^2) - 1)^2
#     @test stress(A, B, ones(n, n)) ≈ s
#     @test distortion(A, B) ≈ 0.1
# end

# @testset "align" begin
#     A = [0.0 0.0;
#          0   1;
#          1   0]
#     # TRANSLATION   
#     B = A .+ 0.1
#     Ahat =  SMACOF.align(B, A)
#     for i in eachindex(Ahat)
#         @test Ahat[i] ≈ A[i] atol = 1e-8
#     end
    
#     # ROTATION
#     B = A * SMACOF.random2Drotation()
#     Ahat =  SMACOF.align(B, A)
#     for i in eachindex(Ahat)
#         @test Ahat[i] ≈ A[i] atol = 1e-8
#     end
# end

# function recover_noiseless_configuration(f, n=50)
#     X = rand(n, 2)
#     Δ = dists(X)
#     Xhat = f(Δ, X)
#     @test SMACOF.mse(X, Xhat) < 1e-6
# end

# @testset "smacof" begin
#     for n in [10, 20, 30, 40, 50]
#         recover_noiseless_configuration((Δ, anchors) -> fit(Smacof(Δ), anchors=anchors), n)
#     end
# end

# @testset "da smacof" begin
#     for n in [10, 20, 30, 40, 50]
#         recover_noiseless_configuration((Δ, anchors) -> da_smacof(Δ, anchors=anchors), n)
#     end
# end

# @testset "wda smacof" begin
#     for n in [10, 20, 30, 40, 50]
#         recover_noiseless_configuration((Δ, anchors) -> wda_smacof(Δ, anchors=anchors), n)
#     end
# end

function recover_noiseless_configuration_partial(f, n=50)
    X = rand(n, 2)
    Δ = dists(X)
    anchors = (idx = 1:5, pos = X[1:5, :])
    Xhat = f(Δ, anchors)
    @test SMACOF.mse(X, Xhat) < 1e-6
end

@testset "smacof" begin
    for n in [10, 20, 30, 40, 50]
        recover_noiseless_configuration_partial((Δ, anchors) -> fit(Smacof(Δ), anchors=anchors), n)
    end
end