"""
    classical_scaling(Δ, p=2)

Given a distance matrix `Δ` and a target dimensionality `p`, find an embedding of points
in p-dimensional Euclidean space such that the between-object distances are preserved as
well as possible.

# Examples
```julia
julia> X = rand(5, 2)
5×2 Matrix{Float64}:
 0.671412  0.326635
 0.145173  0.650203
 0.677262  0.655306
 0.070194  0.4018
 0.507032  0.0532854

julia> Δ = SMACOF.dists(X)
5×5 Matrix{Float64}:
 0.0       0.617757  0.328723  0.605899  0.318969
 0.617757  0.0       0.532114  0.259472  0.698035
 0.328723  0.532114  0.0       0.657873  0.625626
 0.605899  0.259472  0.657873  0.0       0.558829
 0.318969  0.698035  0.625626  0.558829  0.0

julia> Y = classical_scaling(Δ)
5×2 Matrix{Float64}:
  0.271143  -0.0296369
 -0.34334   -0.0931504
  0.133963  -0.328368
 -0.303247   0.163206
  0.241481   0.28795

julia> SMACOF.dists(Y)
5×5 Matrix{Float64}:
 0.0       0.617757  0.328723  0.605899  0.318969
 0.617757  0.0       0.532114  0.259472  0.698035
 0.328723  0.532114  0.0       0.657873  0.625626
 0.605899  0.259472  0.657873  0.0       0.558829
 0.318969  0.698035  0.625626  0.558829  0.0
```
"""
function classical_scaling(Δ, p = 2)
    n = size(Δ, 1)

    # Compute A = - Δ^2 / 2
    A = Hermitian(similar(Δ), :L)
    for j in 1:n, i in j:n
        A.data[i, j] = -Δ[i, j]^2 / 2
    end

    # Doubly center A so that the row and col sums are zero
    ravg = mean(A, dims = 2)
    tavg = mean(ravg)
    for j in 1:n, i in j:n
        A.data[i, j] = A[i, j] - ravg[i] - ravg[j] + tavg
    end

    # compute low rank approximation
    Λ, V = eigen!(A, (n - p + 1):n)
    reverse!(V, dims = 2)
    reverse!(Λ)

    for j in 1:p, i in 1:n
        V[i, j] *= sqrt(abs(Λ[j]))
    end

    return V
end