"""
Dont need preconditioning for thin matrices
"""
mutable struct ConjugateGradient{T <: AbstractFloat}
    r
    r2
    d
    z
    α::T
    β::T
    B
    Vdot
    ε::T
    itmax::Int
    p::Int
    n::Int
    np::Int
    function ConjugateGradient(D, Δk, W, ε, itmax, p, n)
        # Initialize B matrix
        n = size(D, 1)
        B = zeros(n, n)
        for j in 1:n, i in 1:(j - 1)
            B[i, j] = - W[i, j] * Δk[i, j] / D[i, j]
            if B[i, j] > 1e-8
                println("$i $j $(B[i,j])")
            end
        end
        B = Hermitian(B, :U)
        B.data[diagind(B)] = - sum(B, dims=2)

        # Initialize Vdot
        Vdot = - Matrix(W)
        for i in 1:n
            Vdot[i, i] = 1 + sum(W[1:end .!= i, i])
        end
        
        return new{typeof(ε)}(zeros(n, p), zeros(n, p), zeros(n, p), zeros(n, p), 
                0.0, 0.0, B, Vdot, ε, itmax, p, n, n * p)
    end
end

function iterate!(X, k, preconditioner, C::ConjugateGradient)
    # L = preconditioner.L
    L = I(size(X[1], 1))
    C.r = C.B * X[k - 1]
    # C.r = BLAS.symm('L', 'U', C.B, X[k - 1])                # r <- B * X - V * X'
    BLAS.symm!('L', 'L', -1.0, C.Vdot, X[k], 1.0, C.r)
    C.z[:] = L' \ (L \ C.r)                                 # z <- M^-1 * r
    BLAS.blascopy!(C.np, C.z, 1, C.d, 1)                    # d <- r
    for t in 1:C.itmax
        @show norm(C.r)
        if norm(C.r) < C.ε
            println("CG finished after $t iterations")
            break
        end
        # norm(C.r)  < C.ε && break
        C.α = dot(C.r, C.z) / dot(C.d, C.Vdot, C.d)
        axpy!(C.α, C.d, X[k])                               # X  <- X + α d
        C.r2 = C.r - C.α * C.Vdot * C.d                     # Remaining error
        C.z[:] = L' \ (L \ C.r2)                            # z <- M^-1 * r
        C.β = dot(C.r2, C.z) / dot(C.r, C.z)                # New direction
        C.d = C.z + C.β * C.d                               # Direction to move
        C.r = C.r2
    end
end
