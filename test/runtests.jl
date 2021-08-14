using SMACOF
using Test
using Distances
using LinearAlgebra
using Random
using Statistics
Random.seed!(1993)

ϵ = 1e-8

@testset "classical scaling" begin
    X = rand(50, 2)
    Δ = SMACOF.dists(X)
    X̂ = classical_scaling(Δ)
    Δ̂ = SMACOF.dists(X̂)
    @test norm(Δ - Δ̂) < ϵ
end

@testset "smacof float32" begin
    X = rand(Float32, 50, 2)
    Δ = SMACOF.dists(X)
    # Y = fit(Smacof(Δ), anchors = X)
    Y = smacof(Δ, anchors = X)
    @test SMACOF.mse(Y, X) < ϵ
end

@test 1 == 0

@testset "Procrustes" begin
    A = [0.0 0.0;
         0   1;
         1   0]
    # TRANSLATION   
    B = A .+ 0.1
    Ahat =  SMACOF.align(B, A)
    for i in eachindex(Ahat)
        @test Ahat[i] ≈ A[i] atol = 1e-8
    end
        
    # ROTATION
    B = SMACOF.rotate(A)
        Ahat =  SMACOF.align(B, A)
    for i in eachindex(Ahat)
        @test Ahat[i] ≈ A[i] atol = 1e-8
    end

    p = SMACOF.Procrustes(B, A)
    Ahat = SMACOF.transform(B, p)
    @test norm(A - Ahat) ≈ 0.0 atol = 1e-8
end
    

    @testset "recover noiseless configuration" begin
    nlst = [10, 20, 30]
    X = Dict(n => rand(n, 2) for n in nlst)
    Δ = Dict(n => dists(X[n]) for n in nlst)
    ϵ = 1e-6
    
    @testset "smacof" begin
    for n in nlst
            Xhat = fit(Smacof(Δ[n]), anchors = X[n])
            @test SMACOF.mse(X[n], Xhat) < ϵ
        end
    end

            @testset "da_smacof" begin
        for n in nlst
        Xhat = da_smacof(Δ[n], anchors = X[n])
            @test SMACOF.mse(X[n], Xhat) < ϵ
        end
    end

            @testset "wda_smacof" begin
        for n in nlst
            Xhat = wda_smacof(Δ[n], anchors = X[n])
            @test SMACOF.mse(X[n], Xhat) < ϵ
        end
    end
end
            
@testset "noiseless configuration with some anchors" begin
    nlst = [10, 20, 30]
    n_anchors = 5
    X = Dict(n => rand(n, 2) for n in nlst)
    anchors = Dict(n => (idx = 1:n_anchors, pos = X[n][1:n_anchors, :]) for n in nlst)
    Δ = Dict(n => dists(X[n]) for n in nlst)
    ϵ = 1e-6
    
    @testset "smacof" begin
    for n in nlst
    Xhat = fit(Smacof(Δ[n]), anchors = anchors[n])
        @test SMACOF.mse(X[n], Xhat) < ϵ
        end
    end

    @testset "da_smacof" begin
        for n in nlst
        Xhat = da_smacof(Δ[n], anchors = anchors[n])
            @test SMACOF.mse(X[n], Xhat) < ϵ
        end
    end

    @testset "wda_smacof" begin
        for n in nlst
            Xhat = wda_smacof(Δ[n], anchors = anchors[n])
            @test SMACOF.mse(X[n], Xhat) < ϵ
        end
    end
            end
            
# TODO: @test make_gif()
#       makegif returns true if successful