"""
    wda_smacof(Δ, W = nothing; Xinit = nothing, η = 0.9, p = 2, ε = 1e-6, Tmin = 1e-6, 
                anchors = nothing, verbose = false, DA_itmax = 200, return_history = false)

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
function wda_smacof(Δ, W = nothing; Xinit = nothing, η = 0.9, p = 2, ε = 1e-6, Tmin = 1e-6, κ = 0.9,
    anchors = nothing, verbose = false, itmax = 100, DA_itmax = 100, return_history = false)
    # Use uniform weights if left unspecified
    n = size(Δ, 1)
    if isnothing(W)
        W = ones(n, n) - I
    else
        # normalize weights
        C = sum(tril(W .* Δ.^2, -1)) * 2 / (n * (n - 1))
        W = W / C
    end

    # Compute largest T such that Δ has at least one nonzero element
    DA = DeterministicAnnealing(Δ, W, Tmin = Tmin, p = p, η = η, κ = κ, itmax = DA_itmax)

    # Pick random initial mapping
    X = Dict(1 => ifelse(isnothing(Xinit), rand(n, p), Xinit))
    Dk = dists(X[1])
    σ = [stress(Dk, Δ, W)]
    verbose && println("$(DA.k) \t $(round(σ[end], digits = 6))")
    
    # Conjugate Gradient method to solve  `Bdot = B × X = Vdot × Xk`   for X.
    CG = ConjugateGradient(Dk, DA.Δk, W, ε, p, p, n)
    # CG.Vdot = pinv(CG.Vdot)
    while nextiter(DA)
        updateB!(CG, Dk, DA.Δk, W)
        # verbose && @show CG.Vdot[1, 1:5]
        # verbose && @show CG.B[1, 1:5]
        # verbose && @show Dk[1, 1:5]        
        # verbose && @show DA.Δk[1, 1:5]        
        # verbose && @show sum(CG.B)        
        X[DA.k] = deepcopy(X[DA.k - 1])
        iterate!(X, DA.k, CG, verbose)
        # X[DA.k] = CG.Vdot * CG.B * X[DA.k - 1]
        Dk = dists(X[DA.k])
        push!(σ, stress(Dk, DA.Δk, W))
        verbose && println("$(DA.k) \t $(DA.Tk) \t $(round(σ[end], digits = 12))")
        absolute_error(σ) < ε && break
        # verbose && @show Dk[1, 1:10]
        # verbose && @show DA.Δk[1, 1:10]
        updateTk!(DA)
        updateΔk!(DA, Δ, W)
        # updateΔk!(DA.Δk, Δ, W, DA.Tk, DA.p)
        # verbose && println("$(DA.k) \t $(σ[end])")
    end
    Y = fit(Smacof(Δ, Xinit = X[DA.k], W = W, ε = ε, itmax = itmax, verbose = verbose), anchors = anchors)
    return_history && return Y, X
    # verbose && @show Dk[1, 1:5]
    # verbose && @show DA.Δk[1, 1:5]
    # verbose && @show Δ[1, 1:5]
    # verbose && @show X[DA.k][2,:]
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
    B
    Vdot
    ε::T
    itmax::Int
    p::Int
    n::Int
    np::Int
    function ConjugateGradient(D, Δk, W, ε, itmax, p, n)
        # Initialize B matrix
        n = size(D, 1)
        B = Hermitian(zeros(n, n), :U)
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

function updateB!(CG::ConjugateGradient, Dk, Δk, W)
    for j in 1:size(Δk, 1), i in 1:(j - 1)
        if Dk[i, j] < 1e-8 || W[i,j] < 1e-8 
            CG.B.data[i,j] = 0
            continue
        end
        CG.B.data[i, j] = - W[i, j] * Δk[i, j] / Dk[i, j]
    end
    CG.B.data[diagind(CG.B)] .= 0
    CG.B.data[diagind(CG.B)] = - sum(CG.B, dims = 2) 
end