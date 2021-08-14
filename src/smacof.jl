mutable struct Smacof{T <: AbstractFloat}
    Δ               # Dissimilarities
    D               # Distance matrix for best estimate
    Xhist           # All past configurations
    W               # Weight matrix    
    Vinv::Hermitian # Inverse of Weighted Laplacian
    B::Hermitian
    σ               # Stress
    ε::T            # Error tolerance
    n::Int
    bestit::Int
    itmax::Int       
    verbose::Bool
    function Smacof(Δ; W = nothing, Xinit = nothing, p = 2, ε = 1e-12, itmax = 100, verbose = false)
        if !issymmetric(Δ)
            print("First argument is not a dissimilarity matrix. ")
            return
        end
        T = typeof(ε)
        n = size(Δ, 1)
        W = initweights(Δ, W)
        Xhist = Dict(1 => ifelse(isnothing(Xinit), classical_scaling(Δ), Xinit))
        D = dists(Xhist[1])
        B = Hermitian(zeros(n, n), :U)        
        σ = [stress(Δ, D, W)]
        Vinv = getVinv(W)
        verbose && println("1\t stress = ", σ[1])
        return new{T}(Δ, D, Xhist, Matrix{T}(W), Vinv, 
                B, σ, ε, size(Δ, 1), itmax, itmax, verbose)
    end
end

function fit(sm::Smacof; anchors = nothing)
    for i in 2:sm.itmax
        # update_bmat!(sm)
        updateB!(sm.B, sm.Δ, sm.D, sm.W)
        sm.Xhist[i] = sm.Vinv * sm.B * sm.Xhist[i - 1]      # Gutman transform
        sm.D = dists(sm.Xhist[i])
        push!(sm.σ, stress(sm.Δ, sm.D, sm.W))
        sm.verbose && println("$i\t stress = ", sm.σ[i])
        absolute_error(sm.σ) < sm.ε && break
    end
    isnothing(anchors) && return best(sm)
    return SMACOF.align(best(sm), anchors) 
end

function update_bmat!(sm::Smacof)
    for j in 1:sm.n, i in 1:(j - 1)
        sm.W[i, j] ≈ 0.0 && continue
        sm.B.data[i,j] = ifelse(sm.D[i,j] ≥ sm.ε, - sm.W[i,j] * sm.Δ[i,j] / sm.D[i,j], 0.0)
    end
    sm.B[diagind(sm.B)] .= 0.0
    sm.B[diagind(sm.B)] = -sum(sm.B, dims = 2)
end

best(sm::Smacof) = sm.Xhist[length(sm.σ)]
