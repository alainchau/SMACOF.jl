using PyCall
skl = pyimport("sklearn.manifold")
function skl_smacof(Δ; anchors = nothing, eps = ϵ / 2)
    Yhat = skl.smacof(Δ, eps = eps)[1]
    return align(Yhat, anchors)
end

# @testset "misc types" begin
#     X = 3 * rand(Float32, 500, 2)
#     Δ = dist(X)
#     # params = SmacofParams(Δ, anchors = X)
#     @time Y = smacof(Δ, anchors = X)
#     @show mse(Y, X)
#     @test mse(Y, X) < ϵ

#     X = 3 * rand(Float32, 500, 2)
#     Δ = dist(X)
#     # params = SmacofParams(Δ, anchors = X)
#     @time Y = smacof(Δ, anchors = X)
#     @test mse(Y, X) < ϵ
# end

# @testset "regular float 64" begin
#     for i in 1:2
#         X = 3 * rand(500, 2)
#         Δ = dist(X)

#         @time Y = smacof(Δ, anchors = X)
#         @show mse(Y, X)

#         @time Yhat = skl_smacof(Δ, anchors = X)
#         @show mse(Yhat, X)
#     end
# end

@testset "embedded square" begin
    for i in 1:2
        X = rand(500, 3)
        X[:, 3] .= 0.0
        U = X[:, 1:2]
        X = rotate(X)
        Δ = dist(X)
        @time Y = smacof(Δ, anchors = U, itmax = 1000)
        @test stress(Δ, dist(Y)) < ϵ
        
    end
end