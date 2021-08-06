using SMACOF
using Test
using Distances
using LinearAlgebra
using Random
using Statistics
using Plots
ENV["GKSwstype"] = "100"

Random.seed!(1993)

# @testset "stress/distortion" begin
#     A = [0.0    0   1;  0.0    1   0]
#     B = [0.0    0   1;  0.1    1   0]
#     n = size(A, 2)
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
    
# @testset "align" begin
#     A = randn(50, 2)
#     D = dists(A) + rand(50, 50) * 0.1
#     B = classical_mds(D)
#     Ahat =  SMACOF.align(B, A)
#     @test SMACOF.mse(A, Ahat) < 0.001
# end

@testset "smacof" begin
    X = randn(50, 2)
    Y = (X .- mean(X, dims=1)) * SMACOF.random2Drotation()
    sm = Smacof(dists(Y), verbose=false, itmax=200)
    @time Y = fit(sm, anchors=X)
    @test SMACOF.mse(X, Y) < 1e-8
end

# @testset "gethist check size" begin
#     X = [0.0    0   1; 0.0    1   0]
#     Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
#     sm = Smacof(X, verbose=false)
#     Y = fit(sm, anchors=X)
#     @test size(gethist(sm)) == (sm.it[1], 2, 3)
# end

# test_xy(X, Y, atol=1e-8) = @test norm(X - Y) ≈ 0 atol = atol

# @testset "DA smacof" begin
#     X = [0.0    0   1;  0.0    1   0]
#     Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
#     Y = da_smacof(dists(Y), anchors=X)
#     @test norm(X - Y) ≈ 0 atol = 1e-8

#     for i in 1:5
#         X = randn(2, 100)
#         Y = da_smacof(dists(X), verbose=false, anchors=X)
#         # @test norm(X - Y) ≈ 0 atol = 1e-8
#         test_xy(X, Y)
#     end
# end

# @testset "WDA SMACOF" begin
#     # X = [0.0    0   1 2;  0.0    1   0 4]
#     # Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
#     # @time Y = wda_smacof(dists(Y), η=0.95, verbose=true, anchors=X, ε=1e-12)
#     # @test norm(X - Y) ≈ 0 atol = 1e-8

#     # for i in 1:5
#     #     X = randn(2, 100)
#     #     Y = da_smacof(dists(X), verbose=false, anchors=X)
#     #     # @test norm(X - Y) ≈ 0 atol = 1e-8
#     #     test_xy(X, Y)
#     # end

#     # X = rand(2, 10)
#     # Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
#     # @time Y = wda_smacof(dists(Y), η=0.95, verbose=true, anchors=X, ε=1e-12)
#     # @test norm(X - Y) ≈ 0 atol = 1e-8
#     X = rand(200, 2)
#     Y =  (X .- mean(X, dims=1)) * SMACOF.random2Drotation()
#     @time Y = wda_smacof(dists(Y), η=0.8, verbose=true, anchors=X, ε=1e-12)
#     @test SMACOF.mse(X, Y) < 1e-8
# end