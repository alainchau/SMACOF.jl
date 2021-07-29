using SMACOF
using Test
using Distances
using LinearAlgebra
using Random
using Statistics

@testset "stress/distortion" begin
    A = [0.0    0   1; 
         0.0    1   0]
    B = [0.0    0   1; 
         0.1    1   0]
    n = size(A, 2)
    s = (0.9 - 1)^2 + (sqrt(1 + 0.1^2) - 1)^2
    @test stress(A, B, ones(n, n)) ≈ s
    @test distortion(A, B) ≈ 0.1
end

@testset "align" begin
    A = [0.0    0   1; 
         0.0    1   0]
    
    # TRANSLATION   
    B = A .+ 0.1
    Ahat =  SMACOF.align(B, A)
    for i in eachindex(Ahat)
        @test Ahat[i] ≈ A[i] atol = 1e-8
    end
    
    # ROTATION
    B = SMACOF.random2Drotation() * A
    Ahat =  SMACOF.align(B, A)
    for i in eachindex(Ahat)
        @test Ahat[i] ≈ A[i] atol = 1e-8
    end
end

# @testset "smacof translate" begin
#     X = [0.0    0   1; 
#          0.0    1   0]
#     Y = X .+ 0.1 
#     Y = smacof(pairwise(Euclidean(), Y, dims=2))    
#     Y = SMACOF.align(Y, X)
#     @test norm(X - Y) ≈ 0 atol = 1e-8
# end

# @testset "smacof scaling" begin
#     X = [0.0    0   1; 
#          0.0    1   0]
#     Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
#     Y = smacof(pairwise(Euclidean(), Y, dims=2))    
#     Y = SMACOF.align(Y, X)
#     @test norm(X - Y) ≈ 0 atol = 1e-8
    # end
    
@testset "smacof struct" begin
    X = [0.0    0   1; 
         0.0    1   0]
    Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
    sm = Smacof(X)
    Y = fit(sm, anchors=X)
    @test norm(X - Y) ≈ 0 atol = 1e-8
end

@testset "gethist check size" begin
    X = [0.0    0   1; 
    0.0    1   0]
    Y = SMACOF.random2Drotation() * (X .- mean(X, dims=2))
    sm = Smacof(X)
    Y = fit(sm, anchors=X)
    @test size(gethist(sm)) == (3, 2, 3)
end

# @testset "SMACOF.jl" begin
#     n = 1000
#     noise = 0.1

#     Ufull = rand(3, n)
#     Ufull[3,:] .= 0
#     U = Ufull[1:2, :] + randn(2, n) * noise
#     Vpre = SMACOF.random3Drotation() * Ufull
#     D = pairwise(Euclidean(), U, dims=2)
#     V = smacof(D, D .< Inf, init_config=Vpre)
#     Valigned = SMACOF.align(V, U)
#     println(distortion(U, Valigned))
# end