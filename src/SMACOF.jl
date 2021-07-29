module SMACOF

using Distances
using LinearAlgebra
import MultivariateStats: transform, fit
using MultivariateStats
import Statistics.mean
include("procrustes.jl")
include("helpers.jl")

export Smacof, fit, stress, distortion, classical_mds, gethist, dists

struct Smacof
    Δ           # Dissimilarities
    X           # Current best estimate
    D           # Distance matrix for best estimate
    Xhist       # All past configurations
    W           # Weight matrix    
    V
    Vinv
    b
    lb
    ε           # Error tolerance
    it
    itmax       
    verbose
    function Smacof(Δ; W=nothing, X=nothing, ε=1e-12, itmax=10, verbose=false)
        if !issymmetric(Δ)
            if verbose
                print("First argument is not a dissimilarity matrix. ")
                println("Constructing Euclidean distance matrix.")
            end
            Δ = dists(Δ)
        end
        if isnothing(W) 
            W = ones(size(Δ))
            W[diagind(W)] .= 0
        end
        if isnothing(X)
            X = classical_mds(Δ)
        end
        Xhist = zeros(itmax, size(X)...)
        Xhist[1,:,:] = X
        Δ = Δ / sqrt(sum(W .* Δ.^2))        # Normalize
        V = - Matrix{Float64}(W)
        V[diagind(V)] = - sum(V, dims=1)    # Row sums
        Vinv = pinv(V)
        
        D = pairwise(Euclidean(), X, dims=2)
        lb = sum(W .* D .* Δ) / sum(W .* D.^2)
        b = zeros(size(W))
        return new(Δ, X * lb, D * lb, Xhist, W, V, Vinv, b, lb, ε, [1], itmax, verbose)
    end
end

function update_bmat!(sm::Smacof)
    for i in eachindex(sm.b)
        sm.b[i] = sm.D[i] < sm.ε
        sm.b[i] = sm.W[i] * sm.Δ[i] * (1 - sm.b[i]) / (sm.D[i] + sm.b[i])
    end
    for i in 1:size(sm.b, 1)
        sm.b[i,i] = 0
        for j in 1:size(sm.b, 1)
            i == j && continue
            sm.b[i,i] = sm.b[i,i] + sm.b[i,j]
            sm.b[i,j] = - sm.b[i,j]
        end
    end
end

function fit(sm::Smacof; anchors=nothing)
    for i in 2:sm.itmax
        # sm.b = bmat(sm.Δ, sm.W, sm.D, sm.b)
        update_bmat!(sm)
        # e = sm.X * sm.b * sm.Vinv - sm.X
        sm.X[:] = sm.X * sm.b * sm.Vinv
        sm.Xhist[i,:,:] = sm.X / sm.lb
        e = sm.Xhist[i,:,:] - sm.Xhist[i - 1,:,:]
        sm.D[:] = pairwise(Euclidean(), sm.X, dims=2)
        if (sqrt(sum(sm.V .* (e' * e))) < sm.ε) 
            sm.it[1] = i
            break
        end
    end
    if !isnothing(anchors)
        return SMACOF.align(sm.X / sm.lb, anchors) 
    else
        return sm.X / sm.lb
    end
end

gethist(sm::Smacof) = sm.Xhist[1:sm.it[1], :, :]

end