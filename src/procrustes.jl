function procrustes(X, Y)
    mX = mean(X, dims=2)
    mY = mean(Y, dims=2)
    Xb, Yb = X .- mX, Y .- mY
    s = svd(Xb * Yb')
    R = s.V * s.U'
    R, mY - R * mX
end

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
    # test
end

transform(p::Procrustes, X) = p.R * X .+ p.t

function align(xs, ys)
    p = Procrustes(xs, ys)
    return transform(p, xs)
end