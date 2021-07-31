"""
    da_computeT0(Δ)

Find largest T_0 such that δij - T sqrt(2p) > 0 for some i,j.
"""
da_computeT0(Δ, p) = maximum(Δ / sqrt(2p)) * 0.99

"""
    da_updateΔ!(Δ, T, p)

Update Δk with respect to T.
"""
function da_updateΔ!(Δ, Δk, T, p) 
    for i in eachindex(Δ)
        Δk[i] = max(0, Δ[i] - T * sqrt(2p))
    end
end

"""
    da_smacof(Δ, α)

Apply SMACOF with deterministic annealing. The goal is to reduce the likelihood of converging to a non-optimal local minimum.
The idea of the algorithm is to explore various initial mappings before applying SMACOF. Traditionally, SMACOF uses the classical
MDS solution as the initial mapping.

Inputs
    Δ (Symmetric Matrix) : dissimilarity matrix
    α (Float) : Cooling parameter with support (0,1). Smaller values <=> more entropy and exploration 
    p (Int) : Target dimension, typically 2 or 3 for visualization purposes
    ε (Float) : Stopping condition threshold

Reference
    S. Bae, J. Qiu and G. C. Fox, "Multidimensional Scaling by Deterministic Annealing with Iterative Majorization Algorithm," 
    2010 IEEE Sixth International Conference on e-Science, 2010, pp. 222-229, doi: 10.1109/eScience.2010.45.
    http://dsc.soic.indiana.edu/publications/da_smacof.pdf

"""
function da_smacof(Δ; α=0.9, p=2, ε=1e-5, anchors=nothing, verbose=false)
    Tmin = 1e-8 # Based on the paper, this param is ignored/unimportant
    n = size(Δ, 1)
    Tk = da_computeT0(Δ, p)
    Δk = max.(0, Δ .- Tk * sqrt(2p))
    Xk = rand(p, n)
    σ0 = 0
    while Tk ≥ Tmin
        smk = Smacof(Δk, Xinit=Xk)
        Xk = fit(smk)
        σ1 = stress(smk)
        verbose && println("stress = ", σ1)
        if abs(σ1 - σ0) / σ0 < ε
            break
        end 
        Tk = α * Tk
        da_updateΔ!(Δ, Δk, Tk, p) 
        σ0 = σ1
    end
    verbose && println("Final Xinit = $Xk")
    return fit(Smacof(Δ, Xinit=Xk), anchors=anchors)
end
