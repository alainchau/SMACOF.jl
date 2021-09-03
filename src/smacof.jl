"""
    smacof(Δ; W = nothing, Xinit = nothing, anchors = nothing, p = 2, ε = 1e-12, itmax = 100, verbose = false, return_full = false)

# Arguments
- `Δ`: dissimilarity matrix 

"""
function smacof(Δ; W = nothing, Xinit = nothing, anchors = nothing, p = 2, ε = 1e-12, itmax = 100, verbose = false, return_full = false)
    n = size(Δ, 1)
    
    # Check if Δ is a dissimilarity matrix.
    !issymmetric(Δ) && error("First argument is not a dissimilarity matrix.")

    # Normalize weights
    W = initweights(Δ, W)
    
    # Use classical scaling to get initial mapping if Xinit is not provided
    X = ifelse(isnothing(Xinit), classical_scaling(Δ), Xinit)
    D = distance_matrix(X)
    B = Bmat(Δ, D, W)
    σ = [stress(Δ, D, W)]
    Vinv = getV(W, inv = true)
    for i in 2:itmax
        # Gutman transform
        X = Vinv * B * X
        
        distance_matrix!(D, X)
        push!(σ, stress(Δ, D, W))
        absolute_error(σ) < ε && break
        update!(B, Δ, D, W)
    end

    # Use anchors if provided
    ifelse(isnothing(anchors), X, align(X, anchors))
end
