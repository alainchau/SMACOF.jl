struct Smacof
    Δ           # Dissimilarities
    D           # Distance matrix for best estimate
    Xhist       # All past configurations
    W           # Weight matrix    
    Vinv        # Inverse of Weighted Laplacian
    b
    σ           # Stress
    ε           # Error tolerance
    it
    itmax       
    verbose
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
            W[diagind(W)] .= 0
        end
        Xhist = zeros(p, size(Δ, 1), itmax)
        Xhist[:, :, 1] = ifelse(isnothing(Xinit), classical_mds(Δ), Xinit)
        V = - Matrix{Float64}(W)
        V[diagind(V)] = - sum(V, dims=1)    # Row sums
        Vinv = pinv(V)
        D = dists(Xhist[:, :, 1])
        b = zeros(size(W))        
        σ = zeros(itmax)
        σ[1] = stress(Δ, D, W)
        verbose && println("1\t stress = ", σ[1])
        return new(Δ, D, Xhist, W, Vinv, b, σ, ε, [itmax], itmax, verbose)
    end
end

function update_bmat!(sm::Smacof)
    for i in eachindex(sm.b)
        t = (sm.D[i] < sm.ε)
        sm.b[i] = - sm.W[i] * sm.Δ[i]
        sm.b[i] *= (1 - t) / (sm.D[i] + t)
    end
    for i in 1:size(sm.b, 1)
        sm.b[i, i] = -sum(sm.b[i, (1:end) .!= i])
    end
end

function fit(sm::Smacof; anchors=nothing)
    for i in 2:sm.itmax
        update_bmat!(sm)
        sm.Xhist[:, :, i] = sm.Xhist[:, :, i - 1] * sm.b * sm.Vinv
        sm.D[:] = dists(sm.Xhist[:, :, i])
        sm.σ[i] = stress(sm.Δ, sm.D, sm.W)
        sm.verbose && println("$i\t stress = ", sm.σ[i])
        if abs(sm.σ[i - 1] - sm.σ[i]) / sm.σ[i - 1] < sm.ε
            sm.it[1] = i
            break
        end
    end
    if !isnothing(anchors)
        # println(sm.it[1])
        return SMACOF.align(getbest(sm), anchors) 
    else
        return getbest(sm)
    end
end

getbest(sm::Smacof) = sm.Xhist[:, :, sm.it[1]]
gethist(sm::Smacof) = sm.Xhist[:, :, 1:sm.it[1]]
stress(sm::Smacof) = sm.σ[sm.it[1]]