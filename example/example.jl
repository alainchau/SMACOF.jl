using SMACOF
using Plots

function main()
    X = rand(2, 100)

    # Y = X .+ randn(size(X)) * 0.01
    Y = X .+ randn(size(X)) * 0.01

    sm = Smacof(Y)
    Xhat = fit(sm, anchors=X)
    println(sm.it)
    println(gethist(sm))
    # p = scatter(X[1,:], X[2,:], alpha=0.7, markersize=10);
    # scatter!(Xhat[1,:], Xhat[2,:])
    # savefig(p, "example/test.png")
end

main()