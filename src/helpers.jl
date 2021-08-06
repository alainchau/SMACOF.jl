"""
    stress(X, Y, W)

Compute the raw stress.
"""
function stress(DX, DY, W)
    if size(DX, 1) != size(DX, 2)
        DX = pairwise(Euclidean(), DX, dims=1) |> tril
    end
    if size(DY, 1) != size(DY, 2)
        DY = pairwise(Euclidean(), DY, dims=1) |> tril
    end
    
    s = 0.0
    for j in 1:size(W, 1)
        for i in (j + 1):size(W, 2)
            s += W[i, j] * (DX[i, j] - DY[i, j])^2
        end
    end
    return s
end

function stress!(s, k, DX, DY, W)
    if size(DX, 1) != size(DX, 2)
        DX = pairwise(Euclidean(), DX, dims=1) |> tril
    end
    if size(DY, 1) != size(DY, 2)
        DY = pairwise(Euclidean(), DY, dims=1) |> tril
    end
    for j in 1:size(W, 1), i in (j + 1):size(W, 2)
        s[k] += W[i, j] * (DX[i, j] - DY[i, j])^2
    end
end


function distortion(X, Y)
    DX = pairwise(Euclidean(), X, dims=1) |> tril
    DY = pairwise(Euclidean(), Y, dims=1) |> tril
    return  maximum(abs.(DX - DY))
end

"""
    classical_mds(Δ, p=2)

Use classical scaling with dissimilarity matrix Δ.

This method is a simple wrapper for the MultivariateStats library.
"""
classical_mds(Δ, p=2) = copy(transform(fit(MDS, Δ, maxoutdim=p, distances=true))')

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

Get Euclidean distance matrix for X (with columns representing points).
"""
dists(X) = pairwise(Euclidean(), X, dims=1)


random2Drotation() = qr(randn(2, 2)).Q
random3Drotation() = qr(randn(3, 3)).Q

function clear_plots(shape="square_hole")
    for f in readdir("example/gif_stems")
        run(`rm plots/$(shape)/$(f)`)
    end
end

mse(X, Y) = mean(sum((X - Y).^2, dims=2))