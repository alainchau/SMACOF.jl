@testset "regular float 64" begin
    X = 3 * rand(100, 2)
    Δ = dist(X)
    Y = wda_smacof(Δ, anchors = X)


    X = 3 * rand(300, 2)
    Δ = dist(X)
    # @time Y = smacof(Δ, anchors = X)
    # @show mse(Y, X)
    @time Y = wda_smacof(Δ, anchors = X)
    @show mse(Y, X)

    # for i in 1:2
    #     X = 3 * rand(500, 2)
    #     Δ = dist(X)

    #     @time Y = wda_smacof(Δ, anchors = X)
    #     @show mse(Y, X)
    # end
end