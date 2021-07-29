struct Procrustes
    R
    t
    function Procrustes(X, Y)
        mX = mean(X, dims=2)
        mY = mean(Y, dims=2)
        Xb, Yb = X .- mX, Y .- mY
        s = svd(Xb * Yb')
        R = s.V * s.U'
        R, mY - R * mX
        new(R, mY - R * mX) 
    end
end

function align(xs, ys)
    p = Procrustes(xs, ys)
    return p.R * xs .+ p.t
end