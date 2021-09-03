struct Procrustes
    R
    t
    function Procrustes(X, Y)
        # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        mX = mean(X, dims = 1)
        mY = mean(Y, dims = 1)
        Xb, Yb = X .- mX, Y .- mY
        s = svd(Xb' * Yb)
        R = s.U * s.V'
        new(R, mY - mX * R) 
    end
end

function align(X, anchors)
    if typeof(anchors) <: Matrix
        xs = X
        ys = anchors
    else
        xs = X[anchors.idx, :]
        ys = anchors.pos
    end
    p = Procrustes(xs, ys)
    return X * p.R .+ p.t
end

transform(X, p::Procrustes) = X * p.R .+ p.t