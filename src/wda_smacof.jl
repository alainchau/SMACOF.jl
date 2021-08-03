"""
    computeT0(Δ)

Find largest T_0 such that δij - T sqrt(2p) > 0 for some i,j.
"""
computeT0(Δ, p) = maximum(Δ / sqrt(2p)) * 0.99

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

function computeΔ(Δ, W, T, p)
    Δk = zeros(size(Δ))
    updateΔ!(Δ, Δk, W, T, p)
    return Δk
end

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
function wda_smacof(Δ, W=nothing; η=0.9, p=2, ε=1e-8, anchors=nothing, verbose=false, itmax=500, return_history=false)
    # Based on the paper, this param is ignored/unimportant
    Tmin = 1e-8

    # Use uniform weights if left unspecified
    if isnothing(W)
        W = ones(size(Δ))
        W[diagind(W)] .= 0
    end

    # Compute largest T such that Δ has at least one nonzero element
    Tk = computeT0(Δ, p)
    Δk = computeΔ(Δ, W, Tk, p)

    # Pick random initial mapping
    X = zeros(itmax, p, size(Δ, 1))
    X[1, :, :] = randn(size(X[1,:,:]))
    σ0 = Inf
    σ1 = stress(X[1, :,:], Δk, W)

    Vdot = wda_getVdot(W)
    B = wda_getB(dists(X[1,:,:]), Δk, W)
    # Conjugate Gradient method to solve  `Vdot × X = B × Xk`   for X.
    i = 1 
    while (Tk ≥ Tmin) && (i < itmax)
        X[i + 1, :, :] = conjugate_gradient(X[i,:,:], Δk, B, Vdot, W, ε)
        σ0, σ1 = σ1, stress(X[i + 1,:,:], Δk, W)       
        Tk = η * Tk
        updateΔ!(Δ, Δk, W, Tk, p) 
        B = wda_getB(dists(X[i + 1, :, :]), Δk, W, B)
        if abs(σ1 - σ0) < ε
            break
        end
        i += 1
    end
    Y = fit(Smacof(Δ, Xinit=X[i - 1, :, :]), anchors=anchors)
    return_history && return Y, X[1:i, :, :]
    return Y
end

function conjugate_gradient(Xk, Δk, B,  Vdot, W, ε)
    # Random initial points for CG
    Xi = randn(size(Xk))

    # Residual error
    ri = Xk * B - Xi * Vdot
    di = ri
    
    # Converges when residual is small enough
    while norm(ri) > ε
        αi = dot(ri, ri) / dot(di, di * Vdot)   # How much we should move by
        Xi += αi * di                           # Next point
        ri2 = ri - αi * di * Vdot               # Remaining error
        βi = dot(ri2, ri2) / dot(ri, ri)        # New direction
        di = ri2 + βi * di                      # Direction to move
        ri = ri2 
    end
    return Xi
end

function wda_getV(W)
    V = - Matrix{Float64}(W)
    for i in 1:size(V, 1)
        V[i, i] = -sum(W[1:end .!= i, i])
    end
    return V
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

function ruan_dot(Xt, Yt)
    X = Xt'
    Y = Yt'
    s = 0
    for j in 1:size(Y, 2)
        for i in 1:size(Y, 1)
            s += X[j, i] * Y[i, j]
        end
    end
    return s
end