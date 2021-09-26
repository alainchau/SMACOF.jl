"""
    wda_smacof(Δ, W = nothing; Xinit = nothing, η = 0.9, p = 2, ε = 1e-6, Tmin = 1e-6, 
                anchors = nothing, verbose = false, DA_itmax = 200, return_full = false)

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
function wda_smacof(Δ, wt = nothing; Xinit = nothing, η = 0.95, p = 2, ε = 1e-6, Tmin = 1e-12, κ = 0.95, anchors = nothing, verbose = false, itmax = 100, DA_itmax = 100, return_full = false)
    n = size(Δ, 1)

    # Check if Δ is a dissimilarity matrix.
    !issymmetric(Δ) && error("First argument is not a dissimilarity matrix.")

    # Normalize weights
    if isnothing(wt)
        wt = ones(n, n)
    end
    Vdot = - copy(wt)
    for i in 1:size(Vdot, 1)
        Vdot[i, i] = - 1.0
        Vdot[i, i] = -sum(view(Vdot, :, i))
    end
    Δk_idx = findall(wt .≉ 0.0)
    
    # Compute largest T such that Δ has at least one nonzero element
    DA = DeterministicAnnealing(Δ, Δk_idx, Tmin = Tmin, p = p, η = η, κ = κ, itmax = DA_itmax)

    # Pick random initial mapping
    X = Dict(1 => ifelse(isnothing(Xinit), rand(n, p), Xinit))
    Dk = dist(X[1])
    σ = zeros(itmax)
    σ[1] = stress(DA.Δk, Dk, wt)
    
    # Conjugate Gradient method to solve  `Bdot = B × X = Vdot × Xk`   for X.
    CG = ConjugateGradient(Dk, DA.Δk, wt, Vdot, ε, p, p, n)
    while nextiter(DA)
        X[DA.k] = copy(X[DA.k - 1])
        iterate!(X, DA.k, CG, verbose)
        dist!(Dk, X[DA.k])
        updateTk!(DA)
        updateΔk!(DA, Δ, Δk_idx)
        update!(CG.B, DA.Δk, Dk, wt)

        # push!(σ, stress(DA.Δk, Dk, wt))
        σ[DA.k] = stress(DA.Δk, Dk, wt)
        log_output(DA.k - 1, σ, verbose)
        abs(σ[DA.k] - σ[DA.k - 1]) < ε * σ[DA.k - 1] && break
    end
    # if DA.k == DA.itmax
    #     @warn "Algorithm halted because max iterations was reached."
    # end
    println("Finished after $(DA.k) iterations")
    return align(X[DA.k], anchors)
    Y = smacof(Δ, wt = wt, Xinit = X[DA.k], anchors = anchors, ε = ε, itmax = itmax, verbose = verbose)
    return_full && return Dict(:Y => Y, :X => X, :σ => σ)
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
    function DeterministicAnnealing(Δ, Δk_idx; itmax, Tmin, p, η, κ)
        Tk = initializeTk(Δ, Δk_idx, p, κ)
        Δk = zeros(size(Δ))
        updateΔk!(Δk, Δ, Δk_idx, Tk, p)
        return new(Tk, Tmin, Δk, p, 1, itmax, η)
    end
end

function updateΔk!(Δk, Δ, Δk_idx, Tk, p) 
    for i in Δk_idx
        Δk[i] = max(0, Δ[i] - Tk * sqrt(2p))
    end
end

updateΔk!(DA, Δ, Δk_idx) = updateΔk!(DA.Δk, Δ, Δk_idx, DA.Tk, DA.p)
updateTk!(DA) = DA.Tk = DA.Tk * DA.η
iscooled(DA) = DA.Tk < DA.Tmin
ishot(DA) = DA.Tk > DA.Tmin
function nextiter(DA)
    (DA.k == DA.itmax) && return false
    iscooled(DA) && return false
    DA.k += 1
    return true
end

"""
    initializeTk(Δ, wt, p)

Tk needs to be chosen so that Δk has at least one non-zero entry.
"""
function initializeTk(Δ, Δk_idx, p, κ)
    Tk = 0.0
    for i in Δk_idx
        Tk = max(Tk, Δ[i] / sqrt(2p))
    end
    return Tk * κ
end

mutable struct ConjugateGradient{T <: AbstractFloat}
        r
        d
    α::T
    β::T
    B::AbstractMatrix{T}
    Vdot
    ε::T
    itmax::Int
    p::Int
    n::Int
    np::Int
    function ConjugateGradient(D, Δk, W, V, ε, itmax, p, n)
        # Initialize B matrix
        n = size(D, 1)
        B = zeros(typeof(D).parameters[1], size(Δk))
        update!(B, Δk, D, W)
        # V = - Matrix(W)
        # for i in 1:size(V, 1)
        #     V[i,i] = -1.0
        #     V[i, i] = -sum(view(V, :, i))
        # end


        return new{typeof(ε)}(zeros(n, p), zeros(n, p),
                0.0, 0.0, B, V, ε, itmax, p, n, n * p)
    end
end

function iterate!(X, k, C::ConjugateGradient, verbose = false)
    C.r = C.B * X[k - 1] - C.Vdot * X[k]
    C.d = copy(C.r)
    # BLAS.blascopy!(C.np, C.r, 1, C.d, 1)                    # d <- r
    for t in 1:C.itmax
        norm(C.r) < C.ε && break
        rTr = dot(C.r, C.r)
        C.α = rTr / dot(C.d, C.Vdot, C.d)
        X[k] += C.α * C.d
        # axpy!(C.α, C.d, X[k])                               # X  <- X + α d
        C.r -= C.α * C.Vdot * C.d                      # Remaining error
        C.β = dot(C.r, C.r) / rTr                             # New direction
        C.d = C.r + C.β * C.d                               # Direction to move
    end
end