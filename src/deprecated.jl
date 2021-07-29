function bmat(Dhat, W, D, b, eps=1e-12)
    # https://github.com/cran/smacof/blob/40d4994797e71ce7725cb35cef003499705855ec/R/bmat.R
    for i in eachindex(Dhat)
        b[i] = D[i] < eps
        b[i] = W[i] * Dhat[i] * (1 - b[i]) / (D[i] + b[i])
    end
    for i in 1:size(Dhat, 1)
        b[i,i] = 0
        for j in 1:size(Dhat, 1)
            i == j && continue
            b[i,i] = b[i,i] + b[i,j]
            b[i,j] = - b[i,j]
        end
    end
    return b
end


"""
smacof(delta, W; ε=1e-8, itmax=100)

Find points in XY-plane with distance matrix equal to delta. 

Takes (unsquared) distances: delta(i,j) = ||xi - xj||

Based on the R smacofSym algorithm found below:
https://github.com/cran/smacof/blob/40d4994797e71ce7725cb35cef003499705855ec/R/smacofSym.R
"""
function smacof(delta, W=nothing; init_config=nothing, ε=1e-12, itmax=50, return_stress=false, verbose=true)
# Initialize weights
# https://github.com/cran/smacof/blob/40d4994797e71ce7725cb35cef003499705855ec/R/initWeights.R
if isnothing(W) 
    W = ones(size(delta))
    W[diagind(W)] .= 0
end

# If initial guess is not provided, use classical MDS
x_old = ifelse(isnothing(init_config), classical_mds(delta), init_config)

# Normalize delta
dhat = normalize_diss(delta, W)
v = - Matrix{Float64}(W)
v[diagind(v)] = - sum(v, dims=1) # row sums
W = Matrix{Float64}(W)
vinv = pinv(v)
d_old = pairwise(Euclidean(), x_old, dims=2)

# Normalization
lb = sum(W .* d_old .* dhat) / sum(W .* d_old.^2)
x_old = lb * x_old
d_old = lb * d_old
x_new = zeros(size(x_old))
d_new = zeros(size(d_old))
curr_stress = stress(dhat, d_old, W)
b = zeros(size(W))
for i in 1:itmax
    b = bmat(dhat, W, d_old, b)
    x_new = x_old * b * vinv
    d_new = pairwise(Euclidean(), x_new, dims=2)
    curr_stress = stress(dhat, d_new, W)
    verbose && println("Stress = $curr_stress")
    (sqrt(sum(v .* ((x_old - x_new)' * (x_old - x_new)))) < ε) && break
    x_old = x_new
    d_old = d_new
end
# verbose && println("Smacof finished after iterations.")
if return_stress
    return x_old / lb, curr_stress
else
    return x_old / lb
end
end