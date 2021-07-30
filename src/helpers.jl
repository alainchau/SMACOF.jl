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

"""
    classical_mds(Δ, p=2)

Use classical scaling with dissimilarity matrix Δ.

This method is a simple wrapper for the MultivariateStats library.
"""
classical_mds(Δ, p=2) = transform(fit(MDS, Δ, maxoutdim=p, distances=true))



"""
    dists(X)

Get Euclidean distance matrix for X (with columns representing points).
"""
dists(X) = pairwise(Euclidean(), X, dims=2)


random2Drotation() = qr(randn(2, 2)).Q
random3Drotation() = qr(randn(3, 3)).Q

function clear_plots(shape="square_hole")
    for f in readdir("example/gif_stems")
        run(`rm plots/$(shape)/$(f)`)
    end
end
