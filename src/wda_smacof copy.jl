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
function wda_smacof(Δ, W = nothing; Xinit = nothing, η = 0.95, p = 2, ε = 1e-6, Tmin = 1e-12, κ = 0.95, anchors = nothing, verbose = false, itmax = 100, DA_itmax = 100, return_full = false)
    n = size(Δ, 1)

    # Check if Δ is a dissimilarity matrix.
    !issymmetric(Δ) && error("First argument is not a dissimilarity matrix.")

    # Normalize weights
    W = initweights(Δ, W)

    # Compute largest T such that Δ has at least one nonzero element
    DA = DeterministicAnnealing(Δ, W, Tmin = Tmin, p = p, η = η, κ = κ, itmax = DA_itmax)

    # Pick random initial mapping
    X = Dict(1 => ifelse(isnothing(Xinit), rand(n, p), Xinit))
    Dk = dist(X[1])
    # σ = [stress(Δ, Dk, W)]
    σ = [stress(DA.Δk, Dk, W)]
    
    # Conjugate Gradient method to solve  `Bdot = B × X = Vdot × Xk`   for X.
    CG = ConjugateGradient(Dk, DA.Δk, W, ε, p, p, n)
    while nextiter(DA)
        X[DA.k] = copy(X[DA.k - 1])
        iterate!(X, DA.k, CG, verbose)
        Dk = dist(X[DA.k])
        updateTk!(DA)
        updateΔk!(DA, Δ, W)
        update!(CG.B, DA.Δk, Dk, W)

        push!(σ, stress(DA.Δk, Dk, W))
        log_output(DA.k - 1, σ, verbose)
        absolute_error(σ) < ε && break
    end
    if DA.k == DA.itmax
        @warn "Algorithm halted because max iterations was reached."
    end
    Y = smacof(Δ, W = W, Xinit = X[DA.k], anchors = anchors, ε = ε, itmax = itmax, verbose = verbose)
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
    function DeterministicAnnealing(Δ, W; itmax, Tmin, p, η, κ)
        # Tk = maximum(Δ / sqrt(2p)) * κ
        Tk = initializeTk(Δ, W, p, κ)
        Δk = zeros(size(Δ))
        updateΔk!(Δk, Δ, W, Tk, p)
        return new(Tk, Tmin, Δk, p, 1, itmax, η)
    end
end

function updateΔk!(Δk, Δ, W, Tk, p) 
    for i in eachindex(Δ)
        if W[i] ≈ 0.0
            Δk[i] = Δ[i]
        else
            Δk[i] = max(0, Δ[i] - Tk * sqrt(2p))
        end
    end
end

updateΔk!(DA, Δ, W) = updateΔk!(DA.Δk, Δ, W, DA.Tk, DA.p)
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
    initializeTk(Δ, W, p)

Tk needs to be chosen so that Δk has at least one non-zero entry.
"""
function initializeTk(Δ, W, p, κ)
    Tk = 0
    for i in eachindex(Δ)
        if W[i] ≈ 0.0
            continue
        end
        Tk = max(Tk, Δ[i] / sqrt(2p))
    end
    return Tk * κ
end

mutable struct ConjugateGradient{T <: AbstractFloat}
    r
    d
    α::T
    β::T
    B::Bmat
    Vdot
    ε::T
    itmax::Int
    p::Int
    n::Int
    np::Int
    function ConjugateGradient(D, Δk, W, ε, itmax, p, n)
        # Initialize B matrix
        n = size(D, 1)
        # B = Hermitian(zeros(n, n), :U)
        B = Bmat(Δk, D, W)
        # for j in 1:n, i in 1:(j - 1)
        #     B[i, j] = - W[i, j] * Δk[i, j] / D[i, j]
        # end
        # B = Hermitian(B, :U)
        # B.data[diagind(B)] = - sum(B, dims = 2)

        # Initialize Vdot
        # Vdot = - Matrix(W)
        # for i in 1:n
        #     Vdot[i, i] = 1 + sum(W[1:end .!= i, i])
        # end
        V = - Matrix(W)
        V[diagind(V)] = - sum(V, dims = 1)


        return new{typeof(ε)}(zeros(n, p), zeros(n, p),
                0.0, 0.0, B, V, ε, itmax, p, n, n * p)
    end
end

function iterate!(X, k, C::ConjugateGradient, verbose = false)
    C.r = C.B * X[k - 1] - C.Vdot * X[k]
    C.d = deepcopy(C.r)
    # BLAS.blascopy!(C.np, C.r, 1, C.d, 1)                    # d <- r
    for t in 1:C.itmax
        # verbose && @show norm(C.r)
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