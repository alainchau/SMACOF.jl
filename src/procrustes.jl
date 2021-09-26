struct Procrustes
    R
    t
    function Procrustes(X, Y)
        # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        mX = mean(X, dims = 1)
        mY = mean(Y, dims = 1)
        # Xb, Yb = X .- mX, Y .- mY
        s = svd((X .- mX)' * (Y .- mY))
        R = s.U * s.V'
        new(R, mY - mX * R)        
    end
end

function align(X::AbstractMatrix{T}, anchors::AbstractMatrix{T}) where T
    p = Procrustes(X, anchors)
    return X * p.R .+ p.t
end

function align!(X::AbstractMatrix{S}, anchors::AbstractMatrix{T}) where {S,T}
    p = Procrustes(X, anchors)
    X .= X * p.R .+ p.t
    return nothing
end
align!(X, anchors::Nothing) = X

function align(X::AbstractMatrix{T}, anchors::NamedTuple{(:idx, :pos)}) where T
    xs = X[anchors.idx, :]
    ys = anchors.pos
    p = Procrustes(xs, ys)
    return X * p.R .+ p.t
end

function align!(X::AbstractMatrix{T}, anchors::NamedTuple{(:idx, :pos)}) where T
    xs = X[anchors.idx, :]
    ys = anchors.pos
    p = Procrustes(xs, ys)
    X .= X * p.R .+ p.t
end

transform(X, p::Procrustes) = X * p.R .+ p.t