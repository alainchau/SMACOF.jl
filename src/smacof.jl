mutable struct Smacof{T <: AbstractFloat}
    Δ               # Dissimilarities
    D               # Distance matrix for best estimate
    Xhist           # All past configurations
    W               # Weight matrix    
    Vinv            # Inverse of Weighted Laplacian
    b::Hermitian
    σ               # Stress
    ε::T           # Error tolerance
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
        W = getweights(Δ, W)
        isnothing(Xinit) && (Xinit = classical_mds(Δ))
        Xhist = Dict(1 => Xinit)
        D = dists(Xhist[1])
        b = Hermitian(zeros(n, n), :U)        
        σ = [stress(Δ, D, W)]
        Vinv = _smacof_getVinv(W)
        verbose && println("1\t stress = ", σ[1])
        verbose && sum(isnan.(Vinv)) > 0 && @show sum(isnan.(Vinv)) 
        return new{T}(Δ, D, Xhist, Matrix{T}(W), Vinv, 
                b, σ, ε, size(Δ, 1), itmax, itmax, verbose)
    end
end

function fit(sm::Smacof; anchors = nothing)
    for i in 2:sm.itmax
        update_bmat!(sm)
        sm.Xhist[i] = sm.Vinv * sm.b * sm.Xhist[i - 1]      # Gutman transform
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
        sm.b.data[i,j] = ifelse(sm.D[i,j] ≥ sm.ε, - sm.W[i,j] * sm.Δ[i,j] / sm.D[i,j], 0.0)
    end
    sm.b[diagind(sm.b)] .= 0.0
    sm.b[diagind(sm.b)] = -sum(sm.b, dims = 2)
end

best(sm::Smacof) = sm.Xhist[length(sm.σ)]

function _smacof_getVinv(W)
    V = - Matrix{Float64}(W)
    V[diagind(V)] = - sum(V, dims = 1)
    return Hermitian(pinv(V))
end