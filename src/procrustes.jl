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

function align(xs, ys)
    p = Procrustes(xs, ys)
    return xs * p.R .+ p.t
end