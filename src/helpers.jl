"""
    stress(X, Y, W)

Compute the stress.
"""
function stress(Δ, Λ, W = ones(size(Δ)))    
    n = size(Δ, 1)
    s = 0.0
    for j in 1:n, i in (j + 1):n
        s += W[i, j] * (Δ[i, j] - Λ[i, j])^2
    end
    return s
end

"""
    initweights(Δ, W=nothing)

Normalize W so that 
```math
\\sum_{i<j} W[i,j] * Δ[i,j]^2 = \\frac{n * (n-1)}{2}
```
"""
function initweights(Δ, W = nothing)
    n = size(Δ, 1)
    if isnothing(W)
        W = ones(n, n) - I
    end
    C = sum(tril(W .* Δ.^2, -1))
    k = 2 / (n * (n - 1))
    return W / C / k
end


"""
    stress1(X, Y, W)

Compute the stress.
"""
function stress1(Δ, Λ, W)    
    n = size(Δ, 1)
    s = 0.0
    for j in 1:n, i in (j + 1):n
        W[i, j] ≈ 0.0 && continue
        s += W[i, j] * (Δ[i, j] - Λ[i, j])^2
        isnan(s) && error("s is NaN!")
    end
    sqrt(s / sum(tril(W .* Λ.^2, -1)))
end

function stress1(Δ, Λ)    
    n = size(Δ, 1)
    s = 0.0
    sden = 0.0
    for j in 1:n, i in (j + 1):n
        s += (Δ[i, j] - Λ[i, j])^2
        sden += Λ[i, j]^2
        isnan(s) && error("s is NaN!")
    end
    return s / sden
end

relative_error(v) = abs(v[end] - v[end - 1]) / v[end - 1]
absolute_error(v) = abs(v[end] - v[end - 1]) 


"""
    distance_matrix(X)

Get (unsquared) Euclidean distance matrix for X (with rows representing points).
"""
distance_matrix(X) = pairwise(Euclidean(), X, dims = 1)

distance_matrix!(D, X) = pairwise!(D, Euclidean(), X, dims = 1)

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


function getV(W; inv = true)
    V = - copy(W)
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
