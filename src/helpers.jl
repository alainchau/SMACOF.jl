"""
    stress(X, Y, W)

Compute the raw stress.
"""
function stress(X, Y, W)
    DX = pairwise(Euclidean(), X, dims=2) |> tril
    DY = pairwise(Euclidean(), Y, dims=2) |> tril
    return sum(tril(W) .* (DX - DY).^2)   
end

function classical_mds(Δ, p=2)
    transform(fit(MDS, Δ, maxoutdim=p, distances=true))
end