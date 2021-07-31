struct Smacof
    Δ           # Dissimilarities
    X           # Current best estimate
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
    function Smacof(Δ; W=nothing, Xinit=nothing, ε=1e-12, itmax=10, verbose=false)
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
        if isnothing(Xinit)
            Xinit = classical_mds(Δ)
        end
        V = - Matrix{Float64}(W)
        V[diagind(V)] = - sum(V, dims=1)    # Row sums
        Vinv = pinv(V)
        D = dists(Xinit)
        
        b = zeros(size(W))
        Xhist = zeros(itmax, size(Xinit)...)
        Xhist[1,:,:] = Xinit
        σ = zeros(itmax)
        σ[1] = stress(Δ, D, W)
        verbose && println("1\t stress = ", σ[1])
        return new(Δ, Xinit, D, Xhist, W, Vinv, b, σ, ε, [itmax], itmax, verbose)
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
        update_bmat!(sm)
        sm.X[:] = sm.X * sm.b * sm.Vinv
        sm.D[:] = dists(sm.X)
        sm.Xhist[i, :, :] = sm.X
        sm.σ[i] = stress(sm.Δ, sm.D, sm.W)
        sm.verbose && println("$i\t stress = ", sm.σ[i])
        if abs(sm.σ[i - 1] - sm.σ[i]) / sm.σ[i - 1] < sm.ε
            sm.it[1] = i
            break
        end
    end
    if !isnothing(anchors)
        return SMACOF.align(sm.X, anchors) 
    else
        return sm.X
    end
end

gethist(sm::Smacof) = sm.Xhist[1:sm.it[1], :, :]
stress(sm::Smacof) = sm.σ[sm.it[1]]