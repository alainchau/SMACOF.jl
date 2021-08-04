"""
    stress(X, Y, W)

Compute the raw stress.
"""
function stress(DX, DY, W)
    if size(DX, 1) != size(DX, 2)
        DX = pairwise(Euclidean(), DX, dims=2) |> tril
    end
    if size(DY, 1) != size(DY, 2)
        DY = pairwise(Euclidean(), DY, dims=2) |> tril
    end
    
    s = 0.0
    for j in 1:size(W, 1)
        for i in (j + 1):size(W, 2)
            s += W[i, j] * (DX[i, j] - DY[i, j])^2
        end
    end
    return s
end

function stress!(s, DX, DY, W)
    if size(DX, 1) != size(DX, 2)
        DX = pairwise(Euclidean(), DX, dims=2) |> tril
    end
    if size(DY, 1) != size(DY, 2)
        DY = pairwise(Euclidean(), DY, dims=2) |> tril
    end
    for j in 1:size(W, 1)
        for i in (j + 1):size(W, 2)
            s += W[i, j] * (DX[i, j] - DY[i, j])^2
        end
    end
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
