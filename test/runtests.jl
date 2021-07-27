using SMACOF
using Test
using Distances

@testset "Stress" begin
    A = [0.0    0   1; 
         0.0    1   0]
    B = [0.0    0   1; 
         0.1    1   0]
    n = size(A, 2)
    s = (0.9 - 1)^2 + (sqrt(1 + 0.1^2) - 1)^2
    @test stress(A, B, ones(n, n)) ≈ s
end

@testset "SMACOF.jl" begin
    n = 100
    U = rand(2, n)
    Vpre = rand(2, n) + 0.2 * randn(2, n)
    Vpre += randn(size(Vpre)) * 0.2
    D = pairwise(Euclidean(), U, dims=2)
    V = smacof(D, D .< Inf, init_config=Vpre, itmax=10)
    Valigned = SMACOF.align(V, U)
    println(stress(U, V, ones(n, n)))
end