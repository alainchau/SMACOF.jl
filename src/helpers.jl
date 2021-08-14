"""
    stress(X, Y, W)

Compute the stress.
"""
function stress(DX, DY, W)
    if size(DX, 1) != size(DX, 2)
        DX = pairwise(Euclidean(), DX, dims = 1)
    end
    if size(DY, 1) != size(DY, 2)
        DY = pairwise(Euclidean(), DY, dims = 1)
    end

    return _stress(DX, DY, W)
end

function _stress(DX, DY, W)    
    s = 0.0
    for j in 1:size(W, 1)
        for i in (j + 1):size(W, 2)
            W[i, j] ≈ 0.0 && continue
            s += W[i, j] * (DX[i, j] - DY[i, j])^2
            if isnan(s)
                @show W[i, j]
                @show DX[i, j]
                @show DY[i, j]
                error("s is NaN")
            end
        end
    end
    return s
end

function distortion(X, Y)
    DX = pairwise(Euclidean(), X, dims = 1)
    DY = pairwise(Euclidean(), Y, dims = 1)
    return  maximum(abs.(DX - DY))
end

"""
    relative_error(v, i)

Give the relative error of the measurement at i in the vector v.
"""
relative_error(v) = abs(v[end] - v[end - 1]) / v[end - 1]

"""
    absolute_error(v, i)

Give the absolute error of the measurement at i in the vector v.
"""
absolute_error(v) = abs(v[end] - v[end - 1]) 


"""
    dists(X)

Get Euclidean distance matrix for X (with rows representing points).
"""
dists(X) = pairwise(Euclidean(), X, dims = 1)


random2Drotation() = qr(randn(2, 2)).Q
random3Drotation() = qr(randn(3, 3)).Q


function mse(X, Y) 
    ret = mean(sum(skipmissing(X - Y).^2, dims = 2))  
    isnan(ret) && @show sum(isnan.(X))
    isnan(ret) && @show sum(isnan.(Y))
    return ret
end

function mse(X, Y, W) 
    ret = mean(sum(skipmissing(X[W] - Y[W]).^2, dims = 2))  
    isnan(ret) && @show sum(isnan.(X))
    isnan(ret) && @show sum(isnan.(Y))
    return ret
end

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
    get_weights(Δ, W)

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

function getVinv(W)
    V = - Matrix{Float64}(W)
    V[diagind(V)] = - sum(V, dims = 1)
    return Hermitian(pinv(V))
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

# function updateB!(CG::ConjugateGradient, Dk, Δk, W)
#     for j in 1:size(Δk, 1), i in 1:(j - 1)
#         if Dk[i, j] < 1e-8 || W[i,j] < 1e-8 
#             CG.B.data[i,j] = 0
#             continue
#         end
#         CG.B.data[i, j] = - W[i, j] * Δk[i, j] / Dk[i, j]
#     end
#     CG.B.data[diagind(CG.B)] .= 0
#     CG.B.data[diagind(CG.B)] = - sum(CG.B, dims = 2) 
# end