"""
    stress(X, Y, W)

Compute the stress.
"""
function stress(Δ, Λ, W)    
    n = size(Δ, 1)
    s = 0.0
    for j in 1:n, i in (j + 1):n
        W[i, j] ≈ 0.0 && continue
        s += W[i, j] * (Δ[i, j] - Λ[i, j])^2
        isnan(s) && error("s is NaN!")
    end
    s
end

relative_error(v) = abs(v[end] - v[end - 1]) / v[end - 1]
absolute_error(v) = abs(v[end] - v[end - 1]) 


"""
    dists(X)

Get Euclidean distance matrix for X (with rows representing points).
"""
dists(X) = pairwise(Euclidean(), X, dims = 1)

dists!(D, X) = pairwise!(D, Euclidean(), X, dims = 1)

"""
    rotate(A)

Rotate a 2 or 3 dimensional matrix `A`.
"""
function rotate(A)
    p = size(A, 2)
    A * qr(randn(p, p)).Q
end

function mse(X, Y) 
    mean(sum(skipmissing(X - Y).^2, dims = 2))  
end

mse(X, Y, W) = mse(X[W], Y[W])

"""
    issymmetric(A::Matrix{Union{Missing,T}})

Test whether a matrix is symmetric. Ignores `missing` values.
"""
function issymmetric(A::Matrix{Union{Missing,T}}) where {T}
    indsm, indsn = axes(A)
    if indsm != indsn
        return false
    end
    for i in first(indsn):last(indsn), j in (i):last(indsn)
        ismissing(A[i, j]) && continue
        if A[i,j] != transpose(A[j,i])
            return false
        end
    end
    return true 
end

"""
    initweights(Δ, W)

If W is nothing, then return nxn matrix of ones with zeros on the diagonal. 
Otherwise, normalize W so that 
```math
\\sum_{i<j} W[i,j] * Δ[i,j]^2 = \frac{n * (n-1)}{2}
```
"""
function initweights(Δ, W)
    n = size(Δ, 1)
    if isnothing(W)
        W = ones(n, n) - I
    else
        # normalize weights
        C = sum(tril(W .* Δ.^2, -1)) * 2 / (n * (n - 1))
        W = W / C
    end
    return W
end

function getV(W; inv = true)
    V = - Matrix{Float64}(W)
    V[diagind(V)] = - sum(V, dims = 1)
    inv && return Hermitian(pinv(V))
    Hermitian(V)
end

function updateB!(B::Hermitian, Δ, D, W)
    for j in 1:size(Δ, 1), i in 1:(j - 1)
        if D[i, j] < 1e-8 || W[i,j] < 1e-8 
            B.data[i,j] = 0
            continue
        end
        B.data[i, j] = - W[i, j] * Δ[i, j] / D[i, j]
    end
    B.data[diagind(B)] .= 0
    B.data[diagind(B)] = - sum(B, dims = 2) 
end
