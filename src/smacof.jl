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
    it::Int
    itmax::Int       
    verbose::Bool
    function Smacof(Δ; W=nothing, Xinit=nothing, p=2, ε=1e-12, itmax=100, verbose=false)
        if !issymmetric(Δ)
            if verbose
                print("First argument is not a dissimilarity matrix. ")
                println("Constructing Euclidean distance matrix.")
            end
            Δ = dists(Δ)
        end
        if isnothing(W) 
            W = ones(size(Δ))
            W[diagind(W)] .= 0.0
        end
        Xhist = Dict{Int,typeof(Δ)}()
        Xhist[1] = ifelse(isnothing(Xinit), classical_mds(Δ), Xinit)
        D = dists(Xhist[1])
        b = Hermitian(zeros(size(W)), :U)        
        σ = zeros(itmax)
        stress!(σ, 1, D, Δ, W)
        verbose && println("1\t stress = ", σ[1])
        return new{typeof(ε)}(Δ, D, Xhist, W, _smacof_getVinv(W), 
                b, σ, ε, size(Δ, 1), itmax, itmax, verbose)
    end
end

function update_bmat!(sm::Smacof)
    for j in 1:sm.n, i in 1:(j - 1)
        sm.b.data[i,j] = ifelse(sm.D[i,j] ≥ sm.ε, - sm.W[i,j] * sm.Δ[i,j] / sm.D[i,j], 0.0)
    end
    sm.b[diagind(sm.b)] = -sum(sm.b, dims=2) + diag(sm.b)
end

function fit(sm::Smacof; anchors=nothing)
    for i in 2:sm.itmax
        update_bmat!(sm)
        sm.Xhist[i] = sm.Xhist[i - 1] * sm.b * sm.Vinv
        sm.D[:] = dists(sm.Xhist[i])
        stress!(sm.σ, i, sm.Δ, sm.D, sm.W)
        sm.verbose && println("$i\t stress = ", sm.σ[i])
        println(i, "\t", sm.σ[i])
        if (sm.σ[i - 1] - sm.σ[i]) / sm.σ[i - 1] < sm.ε
            sm.it = i
            break
        end
    end
    
    if !isnothing(anchors)
        return SMACOF.align(best(sm), anchors) 
    else
        return best(sm)
    end
end

best(sm::Smacof) = sm.Xhist[sm.it]
stress(sm::Smacof) = sm.σ[sm.it]

function _smacof_getVinv(W)
    V = - Matrix{Float64}(W)
    V[diagind(V)] = - sum(V, dims=1)    # Row sums
    return Hermitian(pinv(V))
end