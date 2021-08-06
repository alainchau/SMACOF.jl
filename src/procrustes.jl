struct Procrustes
    R
    t
    function Procrustes(X, Y)
        # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        mX = mean(X, dims=1)
        mY = mean(Y, dims=1)
        Xb, Yb = X .- mX, Y .- mY
        s = svd(Xb' * Yb)
        R = s.U * s.V'
        new(R, mY - mX * R) 
    end
end

function align(X, anchors)
    xs = X[anchors.idx, :]
    ys = anchors.pos
    p = Procrustes(xs, ys)
    return X * p.R .+ p.t
end