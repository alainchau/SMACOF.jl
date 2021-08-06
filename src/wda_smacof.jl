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
function wda_smacof(Δ, W=nothing; η=0.9, p=2, ε=1e-6, Tmin=1e-6, 
    anchors=nothing, verbose=false, DA_itmax=200, CG_itmax=20, return_history=false)
    # Use uniform weights if left unspecified
    n = size(Δ, 1)
    isnothing(W) && (W = ones(n, n) - I)

    # Compute largest T such that Δ has at least one nonzero element
    DA = DeterministicAnnealing(Δ, W, Tmin=Tmin, p=p, η=η, κ=0.99, itmax=DA_itmax)

    # Pick random initial mapping
    X = Dict(1 => randn(n, p))
    Dk = dists(X[1])
    σ = [stress(Dk, Δ, W)]

    # Conjugate Gradient method to solve  `Vdot × X = B × Xk`   for X.
    CG = ConjugateGradient(Dk, DA.Δk, W, ε, CG_itmax, p, n)
    preconditioner = CholeskyPreconditioner(CG.Vdot, 2)
    while nextiter(DA)
        X[DA.k] = randn(n, p)
        iterate!(X, DA.k, preconditioner, CG)
        Dk = dists(X[DA.k])
        push!(σ, stress(Dk, DA.Δk, W))
        updateTk!(DA)
        updateΔk!(DA, Δ, W)
        updateB!(CG, Dk, DA.Δk, W)
        verbose && println("$(DA.k) \t ", σ[end])
        absolute_error(σ) < ε && break
    end
    Y = fit(Smacof(Δ, Xinit=X[DA.k - 1], ε=ε), anchors=anchors)
    return_history && return Y, X
    return Y
end

mutable struct DeterministicAnnealing
    Tk
    Tmin
    Δk
    p::Int      # Target dimensionality
    k::Int      # Iteration counter
    itmax::Int
    η           # Cooling coefficient
    function DeterministicAnnealing(Δ, W; itmax, Tmin=0.1, p=2, η=0.99, κ=0.99)
        Tk = maximum(Δ / sqrt(2p)) * κ
        Δk = zeros(size(Δ))
        updateΔk!(Δk, Δ, W, Tk, p)
        return new(Tk, Tmin, Δk, p, 1, itmax, η)
    end
end

"""
    updateΔk!(Δ, T, p)

Update Δk with respect to T.
"""
function updateΔk!(Δk, Δ, W, T, p) 
    for i in eachindex(Δ)
        if W[i] ≈ 0.0
            Δk[i] = Δ[i]
        else
            Δk[i] = max(0, Δ[i] - T * sqrt(2 * p))
        end
    end
end

updateΔk!(DA, Δ, W) = updateΔk!(DA.Δk, Δ, W, DA.Tk, DA.p)
updateTk!(DA) = DA.Tk = DA.Tk * DA.η
iscooled(DA) = DA.Tk < DA.Tmin
ishot(DA) = DA.Tk > DA.Tmin
function nextiter(DA)
    DA.k += 1
    (DA.k ≤ DA.itmax) && ishot(DA)
end

mutable struct ConjugateGradient
    r
    r2
    d
    z
    α
    β
    B
    Vdot
    ε
    itmax::Int
    p::Int
    n::Int
    np::Int
    function ConjugateGradient(D, Δk, W, ε, itmax, p, n)
        # Initialize B matrix
        n = size(D, 1)
        B = zeros(n, n)
        for j in 1:n, i in 1:(j - 1)
            B[i, j] = - W[i, j] * Δk[i, j] / D[i, j]
            B[j, i] = B[i, j]
        end
        B[diagind(B)] = - sum(B, dims=2)
    
        # Initialize Vdot
        Vdot = - Matrix{Float64}(W)
        for i in 1:n
            Vdot[i, i] = 1 + sum(W[1:end .!= i, i])
        end

        return new(zeros(n, p), zeros(n, p), zeros(n, p), zeros(n, p), 
                0.0, 0.0, B, Vdot, ε, itmax, p, n, n * p)
    end
end

function iterate!(X, k, preconditioner, C::ConjugateGradient)
    L = preconditioner.L
    C.r = BLAS.symm('L', 'L', C.B, X[k - 1])                # r <- B * X - V * X'
    BLAS.symm!('L', 'L', -1.0, C.Vdot, X[k], 1.0, C.r)
    C.z[:] = L' \ (L \ C.r)                             # z <- M^-1 * r
    BLAS.blascopy!(C.np, C.z, 1, C.d, 1)                    # d <- r
    # threshold = C.ε * norm(C.B * X[k - 1])
    for t in 1:C.itmax
        norm(C.r)  < C.ε && break
        C.α = dot(C.r, C.z) / dot(C.d, C.Vdot, C.d)
        axpy!(C.α, C.d, X[k])                               # X  <- X + α d
        C.r2 = C.r - C.α * C.Vdot * C.d                     # Remaining error
        C.z[:] = L' \ (L \ C.r2)                         # z <- M^-1 * r
        C.β = dot(C.r2, C.z) / dot(C.r, C.z)                # New direction
        C.d = C.z + C.β * C.d                               # Direction to move
        C.r = C.r2
    end
end

function updateB!(CG::ConjugateGradient, Dk, Δk, W)
    for i in 1:size(Δk, 1)
        for j in (i + 1):size(Δk, 1)
            CG.B[i, j] = - W[i, j] * Δk[i, j] / Dk[i, j]
            CG.B[j, i] = CG.B[i, j]
        end
    end
    CG.B[diagind(CG.B)] .= 0
    CG.B[diagind(CG.B)] = - sum(CG.B, dims=2) 
end