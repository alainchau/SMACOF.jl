"""
    wda_smacof(Δ, η)

Apply SMACOF with deterministic annealing with weights. The goal is to reduce the likelihood of converging to a non-optimal local minimum.
The idea of the algorithm is to explore various initial mappings before applying SMACOF. Traditionally, SMACOF uses the classical
MDS solution as the initial mapping.

Inputs
    Δ (Symmetric Matrix) : dissimilarity matrix
    W (Symmetric Matrix) : weight matrix
    η (Float) : Cooling parameter with support (0,1). Smaller values <=> more entropy and exploration 
    p (Int) : Target dimension, typically 2 or 3 for visualization purposes
    ε (Float) : Stopping condition threshold

Reference
    Y. Ruan and G. Fox, "A Robust and Scalable Solution for Interpolative Multidimensional Scaling with Weighting," 
    2013 IEEE 9th International Conference on e-Science, 2013, pp. 61-69, doi: 10.1109/eScience.2013.30.

    http://grids.ucs.indiana.edu/ptliupages/publications/WDA-SMACOF_v1.02.pdf
"""
function wda_smacof(Δ, W=nothing; η=0.9, p=2, ε=1e-8, Tmin=1e-8, anchors=nothing, verbose=false, itmax=500, return_history=false)
    println("ok")
    # Use uniform weights if left unspecified
    if isnothing(W)
        W = ones(size(Δ))
        W[diagind(W)] .= 0
    end

    # Compute largest T such that Δ has at least one nonzero element
    Tk, Δk = initialize_T_and_Δ(Δ, W, p)

    # Pick random initial mapping
    X = randn(p, size(Δ, 1), itmax)
    D = dists(X[:,:,1])
    σ0, σ1 = Inf, stress(D, Δk, W)

    # Conjugate Gradient method to solve  `Vdot × X = B × Xk`   for X.
    CG = ConjugateGradient(wda_getB(D, Δk, W), wda_getVdot(W), ε, size(X, 1), size(X, 2))
    while (Tk ≥ Tmin) && (CG.i < itmax) && abs(σ1 - σ0) > ε
        iterate!(X, CG)
        D = dists(X[:,:,CG.i])
        σ0, σ1 = σ1, stress(D, Δk, W)       
        Tk = η * Tk
        updateΔ!(Δ, Δk, W, Tk, p) 
        wda_getB(D, Δk, W, CG.B)
        verbose && println(σ1)
    end
    Y = fit(Smacof(Δ, Xinit=X[:, :, CG.i], ε=ε), anchors=anchors)
    return_history && return Y, X[:, :, 1:i]
    return Y
end

mutable struct ConjugateGradient
    r
    r2
    d
    i
    α
    β
    B
    Vdot
    ε
    p::Int
    n::Int
    np::Int
    function ConjugateGradient(B, Vdot, ε, p, n)
        return new(zeros(p, n), zeros(p, n), zeros(p, n), 1, 0.0, 0.0, B, Vdot, ε, p, n, n * p)
    end
end

function iterate!(X, C::ConjugateGradient)
    C.r = BLAS.symm('R', 'L', C.B, X[:,:,C.i])                    # r <- X B - X' V
    BLAS.symm!('R', 'L', -1.0, C.Vdot, X[:,:,C.i + 1], 1.0, C.r)
    BLAS.blascopy!(C.np, C.r, 1, C.d, 1)                        # d <- r
    while norm(C.r) > C.ε
        C.α = dot(C.r, C.r) / dot(C.d', C.Vdot, C.d')
        axpy!(C.α, C.d, view(X, :, :, C.i + 1))                           # X  <- X + α d
        C.r2 = C.r - C.α * C.d * C.Vdot               # Remaining error
        C.β = dot(C.r2, C.r2) / dot(C.r, C.r)        # New direction
        C.d = C.r2 + C.β * C.d                      # Direction to move
        C.r = C.r2
    end
    C.i += 1
end

"""
    updateΔ!(Δ, T, p)

Update Δk with respect to T.
"""
function updateΔ!(Δ, Δk, W, T, p) 
    for i in eachindex(Δ)
        if W[i] ≈ 0
            Δk[i] = Δ[i]
        end
        Δk[i] = max(0, Δ[i] - T * sqrt(2p))
    end
end

function initialize_T_and_Δ(Δ, W, p)
    T = maximum(Δ / sqrt(2p)) * 0.99
    Δk = zeros(size(Δ))
    updateΔ!(Δ, Δk, W, T, p)
    return T, Δk
end

function wda_getVdot(W)
    Vdot = - Matrix{Float64}(W)
    for i in 1:size(W, 1)
        Vdot[i, i] = 1 + sum(W[1:end .!= i, i])
    end
    return Vdot
end

function wda_getB(D, Δk, W, B=zeros(size(W)))
    B .= 0
    for i in 1:size(Δk, 1)
        for j in (i + 1):size(Δk, 1)
            B[i, j] = - W[i, j] * Δk[i, j] / D[i, j]
            B[j, i] = B[i, j]
        end
    end
    B[diagind(B)] = - sum(B, dims=2)
    return B
end