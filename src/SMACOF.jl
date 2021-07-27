module SMACOF

using Distances
using LinearAlgebra
import MultivariateStats: transform, fit
using MultivariateStats
using StatsBase
include("procrustes.jl")
include("helpers.jl")

export smacof, stress, classical_mds

function normalize_diss(D, W)
    N = length(D)
    return D / sqrt(sum(W .* D.^2)) * sqrt(N)
end


"""
    smacof(delta, W; ε=1e-8, itmax=100)

Determine configuration of points in XY-plane that have distance matrix equal to delta. 

Takes squared distances: delta(i,j) = ||xi - xj||^2
"""
function smacof(delta, W=nothing; init_config=nothing, ε=1e-12, itmax=50, return_stress=false)
    # Transcribe R algorithm
    # https://github.com/cran/smacof/blob/40d4994797e71ce7725cb35cef003499705855ec/R/smacofSym.R
    # delta = sqrt.(delta)

    # Copy the weight initialization scheme from R smacof
    # https://github.com/cran/smacof/blob/40d4994797e71ce7725cb35cef003499705855ec/R/initWeights.R
    isnothing(W) && (W = ones(size(delta)))

    # If initial guess is not provided, use classical MDS
    x_old = ifelse(isnothing(init_config), classical_mds(delta), init_config)

    n = length(delta)
    nn = n * (n - 1) / 2

    # Normalize delta
    dhat = normalize_diss(delta, W)
    
    # V matrix
    v = - Matrix{Float64}(W)
    v[diagind(v)] = - sum(v, dims=1) # row sums
    W = Matrix{Float64}(W)
    vinv = pinv(v)
    
    d_old = pairwise(Euclidean(), x_old, dims=2)

    # Normalization
    lb = sum(W .* d_old .* dhat) / sum(W .* d_old.^2)
    x_old = lb * x_old
    d_old = lb * d_old

    s_old = sum(W .* (dhat - d_old).^2) / nn 
    e_old = Inf
    itel = 1
    
    while true
        # println(s_old)
        # Get bmat
        # https://github.com/cran/smacof/blob/40d4994797e71ce7725cb35cef003499705855ec/R/bmat.R
        z = replace(x -> x < 1E-12 ? 1 : 0, d_old)
        b = W .* dhat .* (1 .- z) ./ (d_old + z)
        b = diagm(sum(b, dims=1) |> vec) - b
        
        # Get new x
        x_new = x_old * b * vinv
        d_new = pairwise(Euclidean(), x_new, dims=2)
        s_new = sum(W .* (dhat - d_new).^2) / nn
        e_new = sqrt(sum(v .* ((x_old - x_new)' * (x_old - x_new))))
        e_new = abs(e_new)
        r_new = e_new^(1 / itel)
        q_neq = e_new / e_old
        if (e_new < ε) || (itel == itmax)
            break
        end
        x_old = x_new
        d_old = d_new
        s_old = s_new
        e_old = e_new
        itel = itel + 1
    end
    println("Smacof finished after $(itel) iterations.")
    return_stress && return x_old / lb, s_old
    return x_old / lb
end

# smacof_test()



end
