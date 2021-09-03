mutable struct Bmat{T <: AbstractFloat}
    B::Hermitian{T,AbstractMatrix{T}}
    n::Int32
end
function Bmat(n, T = Float64) 
    Bmat{T}(Hermitian(zeros(T, n, n), :U), n)
end
function Bmat(Δ, D, W, T = Float64) 
    n = size(Δ, 1)
    B = Bmat{T}(Hermitian(zeros(T, n, n), :U), n)
    update!(B, Δ, D, W)
    return B
end

getindex(bmat::Bmat, inds...) = bmat.B[inds...]
setindex!(bmat::Bmat, x, inds...) = bmat.B.data[inds...] = x
diagind(bmat::Bmat) = diagind(bmat.B)
sum(bmat::Bmat; dims = dims) = sum(bmat.B, dims = dims)
*(bmat::Bmat, M::AbstractMatrix{T}) where T = bmat.B * M
*(M::AbstractMatrix{T}, bmat::Bmat) where T = M * bmat.B
axes(bmat::Bmat) = axes(bmat.B)
axes(bmat::Bmat, d) = axes(bmat.B, d)
size(bmat::Bmat) = (bmat.n, bmat.n)
function mul!(C::AbstractMatrix{T}, D::AbstractMatrix{T}, B::Bmat) where T 
    mul!(C, D, B.B)
end
function mul!(C::AbstractMatrix{T}, B::Bmat, D::AbstractMatrix{T}) where T 
    mul!(C, B.B, D)
end

function update!(B::Bmat, Δ, D, W)
    for j in 1:B.n, i in 1:(j - 1)
        if D[i, j] < 1e-8
            B[i, j] = 0
            continue
        end
        B[i, j] = - W[i, j] * Δ[i, j] / D[i, j]
    end
    for i in 1:B.n
        B[i, i] = -sum(B[(1:end) .!= i, i])
    end
end