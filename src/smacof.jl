# mutable struct Smacof{S <: Integer,T <: AbstractFloat}
#     Δ::AbstractMatrix{T}                    # Dissimilarities
#     D::AbstractMatrix{T}                    # Distance matrix for best estimate
#     Xhist::Dict{S,AbstractMatrix{T}}        # All past configurations
#     W::AbstractMatrix{T}                    # Weight matrix    
#     Vinv::Hermitian{T,AbstractMatrix{T}}   # Inverse of Weighted Laplacian
#     B::Hermitian{T,AbstractMatrix{T}}
#     σ::AbstractVector{T}                    # Stress
#     anchors::AbstractMatrix{T}
#     ε::T                                    # Error tolerance
#     n::S
#     bestit::S
#     itmax::S
#     verbose::Bool
# end

# function Smacof(Δ::AbstractMatrix{T}; W = nothing, Xinit = nothing, 
#             p = 2, ε = 1e-12, itmax = 100, verbose = false, int_type = Int32, anchors = nothing) where {T}
#     if !issymmetric(Δ)
#         print("First argument is not a dissimilarity matrix. ")
#         return 
#     end
#     n = size(Δ, 1)
#     W = initweights(Δ, W)
#     Xhist = Dict(1 => ifelse(isnothing(Xinit), classical_scaling(Δ), Xinit))
#     D = dists(Xhist[1])
#     B = Hermitian(zeros(n, n), :U)        
#     σ = [stress(Δ, D, W)]
#     Vinv = getVinv(W)
#     verbose && println("1\t stress = ", σ[1])
#     return Smacof{int_type,T}(Δ, D, Matrix{T}(W), Vinv, 
#             B, σ, anchors, ε, size(Δ, 1), itmax, itmax, verbose)
# end

# function fit(sm::Smacof; anchors = nothing)
#     for i in 2:sm.itmax
#         updateB!(sm.B, sm.Δ, sm.D, sm.W)
#         sm.Xhist[i] = sm.Vinv * sm.B * sm.Xhist[i - 1]      # Gutman transform
#         sm.D = dists(sm.Xhist[i])
#         push!(sm.σ, stress(sm.Δ, sm.D, sm.W))
#         sm.verbose && println("$i\t stress = ", sm.σ[i])
#         absolute_error(sm.σ) < sm.ε && break
#     end
#     isnothing(anchors) && return best(sm)
#     return SMACOF.align(best(sm), anchors) 
# end

mutable struct Bmat{T <: AbstractFloat}
    B::Hermitian{T,AbstractMatrix{T}}
    n::Int32
end
function Bmat(n, T = Float64) 
    Bmat{T}(Hermitian(zeros(T, n, n), :U), n)
end

getindex(bmat::Bmat, inds...) = bmat.B[inds...]
setindex!(bmat::Bmat, x, inds...) = bmat.B.data[inds...] = x
diagind(bmat::Bmat) = diagind(bmat.B)
sum(bmat::Bmat; dims = dims) = sum(bmat.B, dims = dims)
*(bmat::Bmat, M::AbstractMatrix{T}) where T = bmat.B * M
*(M::AbstractMatrix{T}, bmat::Bmat) where T = M * bmat.B
axes(bmat::Bmat) = axes(bmat.B)
axes(bmat::Bmat, d) = axes(bmat.B, d)
function mul!(C::Matrix, B::Bmat, D::Matrix) 
    mul!(C, B.B, D, 1.0, 0.0)
end

function update!(B::Bmat, Δ, D, W)
    for j in 1:B.n, i in 1:(j - 1)
        if D[i, j] < 1e-8 || W[i,j] < 1e-8 
            B[i, j] = 0
            continue
        end
        B[i, j] = - W[i, j] * Δ[i, j] / D[i, j]
    end

    for i in 1:B.n
        B[i, i] = -sum(B[(1:end) .!= i, i])
    end
end

function log_output(verbose, σ, i)
    !verbose && return
    println("i\t stress = $(σ[i])")
end

function smacof(Δ::AbstractMatrix{T}; W = nothing, Xinit = nothing, 
    p = 2, ε = 1e-12, itmax = 100, verbose = false, int_type = Int32, 
    anchors = nothing) where {T}
    if !issymmetric(Δ)
        print("First argument is not a dissimilarity matrix. ")
        return 
    end
    n = size(Δ, 1)
    W = initweights(Δ, W)
    Xhist = Dict(1 => ifelse(isnothing(Xinit), classical_scaling(Δ), Xinit))
    D = dists(Xhist[1])
    B = Bmat(n)
    σ = [stress(Δ, D, W)]
    Vinv = getV(W, inv=true) #! weighted laplacian

    log_output(verbose, σ, 1)
    for i in 2:itmax
        update!(B, Δ, D, W)
        Xhist[i] = Vinv * B * Xhist[i - 1]      # Gutman transform
        dists!(D, Xhist[i])
        push!(σ, stress(Δ, D, W))
        log_output(verbose, σ, i)
        absolute_error(σ) < ε && break
    end
    isnothing(anchors) && return last(Xhist)
    return SMACOF.align(last(Xhist), anchors) 
end

last(d::Dict{Int,T}) where {T} = d[maximum(keys(d))]