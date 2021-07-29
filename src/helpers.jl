"""
    stress(X, Y, W)

Compute the raw stress.
"""
function stress(X, Y, W)
    DX = pairwise(Euclidean(), X, dims=2) |> tril
    DY = pairwise(Euclidean(), Y, dims=2) |> tril
    return sum(tril(W) .* (DX - DY).^2)   
end

function distortion(X, Y)
    DX = pairwise(Euclidean(), X, dims=2) |> tril
    DY = pairwise(Euclidean(), Y, dims=2) |> tril
    return  maximum(abs.(DX - DY))
end

function classical_mds(Δ, p=2)
    transform(fit(MDS, Δ, maxoutdim=p, distances=true))
end

normalize_diss(D, W) = D / sqrt(sum(W .* D.^2))

function normalize_diss_n(D, W)
    n = size(D, 1)
    N = n * (n + 1) / 2
    return D / sqrt(sum(W .* D.^2)) * sqrt(N)
end


"""
    dists(X)

Get Euclidean distance matrix for matrix X with columns representing points.
"""
function dists(X)
    return pairwise(Euclidean(), X, dims=2)
end

random2Drotation() = qr(randn(2, 2)).Q
random3Drotation() = qr(randn(3, 3)).Q
