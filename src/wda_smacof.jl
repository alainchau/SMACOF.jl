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
function wda_smacof(Δ, W=ones(size(Δ)); η=0.9, p=2, ε=1e-5, anchors=nothing, verbose=false)
    Tmin = 1e-8 # Based on the paper, this param is ignored/unimportant
    n = size(Δ, 1)
    Tk = computeT0(Δ, p)
    Δk = max.(0, Δ .- Tk * sqrt(2p))
    Xk = rand(p, n)
    σ0 = 0
    σ1 = Inf
    V = wda_getV(W)
    Vdot = wda_getVdot(W)
    B = wda_getB(Δk, Xk, W)
    while Tk ≥ Tmin
        # σ1 = stress(Δ, Δk, W)
        # Conjugate Gradient method
        # init Xi?
        Xi = zeros(size(Xk))
        ri = Xi * B' - Xk * Vdot'
        di = ri
        σi0 = 0
        σi1 = stress(Xi, Δk, W)
        println("----")
        while abs(σi1 - σi0) > ε
            αi = sum(ri * ri') / sum(di * Vdot' * di')
            Xi += αi * di
            ri2 = ri - αi * di * Vdot 
            βi = sum(ri2 *  ri2') / sum(ri * ri')
            di = ri2 + βi * di
            ri = ri2 

            σi0 = σi1
            σi1 = stress(Xi, Δk, W)
            println(Xi)
        end
        # smk =stress(smk)
        println(Xi)
        Xk = Xi
        σ0 = σ1
        σ1 = σi1
        if abs(σ0 - σ1) < ε
            break
        end
        # verbose && println("stress = ", σ1)
        Tk = η * Tk
        updateΔ!(Δ, Δk, W, Tk, p) 
    end
    verbose && println("Final Xinit = $Xk")
    return fit(Smacof(Δ, Xinit=Xk), anchors=anchors)
end

function wda_getV(W)
    V = - Matrix{Float64}(W)
    for i in 1:size(V, 1)
        V[i, i] = -sum(W[1:end .!= i, i])
    end
    return V
end

function wda_getVdot(W)
    V = - Matrix{Float64}(W)
    for i in 1:size(V, 1)
        V[i, i] = 1 + sum(W[1:end .!= i, i])
    end
    return V
end

function wda_getB(Δk, Xk, W, B=zeros(size(W)))
    D = dists(Xk)
    for i in 1:size(Δk, 1)
        for j in (i + 1):size(Δk, 1)
            B[i, j] = - W[i, j] * Δk[i, j] / D[i, j]
            B[j, i] = B[i, j]
        end
    end
    B[diagind(B)] = - sum(B, dims=2)
    return B
end

"""
    ruan_dot(X, Y)

Implementation of dot product from Ruan-Fox paper.
"""
ruan_dot(X, Y) = Y' * X'