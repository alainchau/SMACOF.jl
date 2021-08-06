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
    function Smacof(Δ; W=nothing, Xinit=nothing, p=2, ε=1e-12, itmax=100, verbose=false)
        if !issymmetric(Δ)
            print("First argument is not a dissimilarity matrix. ")
            return
        end
        n = size(Δ, 1)
        if isnothing(W) 
            W = ones(n, n) - I(n)
        end
        isnothing(Xinit) && (Xinit = classical_mds(Δ))
        Xhist = Dict(1 => Xinit)
        D = dists(Xhist[1])
        b = Hermitian(zeros(n, n), :U)        
        σ = [stress(D, Δ, W)]
        verbose && println("1\t stress = ", σ[1])
        return new{typeof(ε)}(Δ, D, Xhist, W, _smacof_getVinv(W), 
                b, σ, ε, size(Δ, 1), itmax, itmax, verbose)
    end
end

function fit(sm::Smacof; anchors=nothing)
    for i in 2:sm.itmax
        update_bmat!(sm)
        sm.Xhist[i] = sm.Vinv * sm.b * sm.Xhist[i - 1]      # Gutman transform
        sm.D = dists(sm.Xhist[i])
        push!(sm.σ, stress(sm.D, sm.Δ, sm.W))
        sm.verbose && println("$i\t stress = ", sm.σ[i])
        absolute_error(sm.σ) < sm.ε && break
    end
    
    !isnothing(anchors) && return SMACOF.align(best(sm), anchors) 
    return best(sm)
end

function update_bmat!(sm::Smacof)
    for j in 1:sm.n, i in 1:(j - 1)
        sm.b.data[i,j] = ifelse(sm.D[i,j] ≥ sm.ε, - sm.W[i,j] * sm.Δ[i,j] / sm.D[i,j], 0.0)
    end
    sm.b[diagind(sm.b)] = -sum(sm.b, dims=2) + diag(sm.b)
end

best(sm::Smacof) = sm.Xhist[length(sm.σ)]

function _smacof_getVinv(W)
    V = - Matrix{Float64}(W)
    V[diagind(V)] = - sum(V, dims=1)
    return Hermitian(pinv(V))
end