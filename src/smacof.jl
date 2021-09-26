function smacof(Δ::AbstractMatrix{T}; p = 2, wt = nothing, Xinit = nothing, anchors = nothing, ε = 1e-6, itmax = 100, verbose = false, return_hist::Bool = false) where T
    n = size(Δ, 1)
    
    # Check if Δ is a dissimilarity matrix.
    !issymmetric(Δ) && error("First argument is not a dissimilarity matrix.")

    # Check if anchors is correct dims
    if !isnothing(anchors)
        if isa(anchors, NamedTuple)
            if size(anchors.pos) != (n, p)
                error("`anchors` has incorrect dimensions. $(size(anchors.pos)) != ($n, $p)")
            end
        else 
            if size(anchors) != (n, p)
                error("`anchors` has incorrect dimensions. $(size(anchors)) != ($n, $p)")
            end
        end
    end

    # Normalize weights
    if isnothing(wt)
        wt = ones(n, n)
        Vinv = I(n) / n
    else
        V = - copy(wt)
        for i in 1:size(V, 1)
            V[i, i] = -sum(view(V, :, i))
        end
        Vinv = pinv(V)
    end
    
    # Use classical scaling to get initial mapping if Xinit is not provided
    X = isnothing(Xinit) ? classical_scaling(Δ, p) : Xinit
    return_hist && (Xhist = [X])
    D = dist(X)
    B = zeros(T, size(Δ))
    update!(B, Δ, D, wt)
    σ = zeros(T, itmax + 1)
    σ[1] = stress(Δ, D, wt)

    Xcache = zeros(size(X))
    for i in 1:itmax
        log_output(i, σ, verbose)
        # Gutman transform
        mul!(Xcache, B, X)
        mul!(X, Vinv, Xcache)
        return_hist && push!(Xhist, copy(X))
        dist!(D, X)
        σ[i + 1] = stress(Δ, D, wt)
        abs(σ[i] - σ[i + 1]) < ε * σ[i] && break
        update!(B, Δ, D, wt)
    end

    # Use anchors if provided
    align!(X, anchors)
    return_hist ? (hist = Xhist, σ = σ) : X
end

function update!(B, Δ, D, W)
    for ij in eachindex(B)
        B[ij] = W[ij] * Δ[ij] / (D[ij] + eps())
    end
        
    for i in 1:size(B, 1)
        B[i, i] = 0.0
        B[i, i] = -sum(view(B, :, i))
    end
end