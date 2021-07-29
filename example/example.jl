using SMACOF
using Plots
gr()
ENV["GKSwstype"] = "100"

function main()
    X = rand(2, 10)

    # Y = X .+ randn(size(X)) * 0.01
    Y = X .+ randn(size(X)) * 0.01

    sm = Smacof(Y)
    Xhat = fit(sm, anchors=X)
    # p = scatter(X[1,:], X[2,:], alpha=0.7, markersize=10);
    # scatter!(Xhat[1,:], Xhat[2,:])
    # savefig(p, "example/test.png")
    for i in 1:sm.it[1]
        p = scatter(X[1,:], X[2,:], alpha=0.7, markersize=10);
        Xi = SMACOF.align(sm.Xhist[i,:,:], X)
        println(distortion(dists(X), dists(Xi)))
        scatter!(Xi[1,:], Xi[2,:]);
        savefig("example/step$i.png");
    end

end

main()