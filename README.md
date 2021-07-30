# SMACOF

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://alainchau.github.io/SMACOF.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alainchau.github.io/SMACOF.jl/dev)
[![Build Status](https://travis-ci.com/alainchau/SMACOF.jl.svg?branch=master)](https://travis-ci.com/alainchau/SMACOF.jl)
[![Coverage](https://codecov.io/gh/alainchau/SMACOF.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/alainchau/SMACOF.jl)

# SMACOF (Scaling by MAjorizing a COmplicated Function)
SMACOF is a multidimensional scaling algorithm which minimizes an objective function (the stress) using a majorization technique. The result is typically a configuration of points in a lower-dimensional space.

Example:
```
using SMACOF
using Plots
gr()
ENV["GKSwstype"] = "100"

function main(n=500)
    # Latent positions to be estimated
    X = rand(2, n)
    # Pick random points to illustrate comvergence
    Xinit = rand(2, n)
    # SMACOF returns a configuration that has the same distance matrix as X
    sm = Smacof(dists(X), Xinit=Xinit, verbose=true, itmax=100)
    fit(sm)

    # Clear image folder
    for f in readdir("example/gif_stems")
        run(`rm example/gif_stems/$(f)`)
    end

    # Make mp4 and then convert to gif
    loadpath = "/home/alain/code/SMACOF/example/gif_stems/"
    animation = Animation(loadpath, String[])
    for i ∈ 1:sm.it[1]
        p = scatter(X[1,:], X[2,:], alpha=0.7, markersize=10, label="true");
        Xi = SMACOF.align(sm.Xhist[i,:,:], X)
        scatter!(Xi[1,:], Xi[2,:], label="smacof");
        xlims!((-0.2, 1.2))
        ylims!((-0.2, 1.2))
        frame(animation, p)
    end
    run(`ffmpeg -y -r 15 -i $loadpath"%06d.png" -vcodec libx264 -crf 25 "anim.mp4"`)
    run(`ffmpeg -y -i anim.mp4 -vf "fps=10,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 anim.gif`)
    run(`rm anim.mp4`)
end

main();
```

![](https://github.com/alainchau/SMACOF.jl/blob/main/anim.gif)
