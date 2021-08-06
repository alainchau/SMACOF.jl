# SMACOF (Scaling by MAjorizing a COmplicated Function)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://alainchau.github.io/SMACOF.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alainchau.github.io/SMACOF.jl/dev)
[![Build Status](https://travis-ci.com/alainchau/SMACOF.jl.svg?branch=main)](https://travis-ci.com/alainchau/SMACOF.jl)
[![codecov](https://codecov.io/gh/alainchau/SMACOF.jl/branch/main/graph/badge.svg?token=eQNUwU70aB)](https://codecov.io/gh/alainchau/SMACOF.jl)

SMACOF is a multidimensional scaling algorithm which minimizes an objective function (the stress) using a majorization technique. The result is typically a configuration of points in a lower-dimensional space.

## Example
```
using SMACOF
using Plots
using Random
Random.seed!(2021)
gr()
ENV["GKSwstype"] = "100"

relpath = "examples/tmp/"
abspath = "/home/alain/code/SMACOF/examples/tmp/"
gifdir = "examples/simple/"
filename = "simple"

X = rand(400, 2)
X = X[.!((1 / 4 .< X[:,1] .< 3 / 4) .& (1 / 4 .< X[:,2] .< 3 / 4)), :]
Xhist = wda_smacof(dists(X), η=0.8, verbose=true, anchors=X, ε=1e-6, return_history=true)[2]

# Make mp4 and then convert to gif
animation = Animation(abspath, String[])
lst_mse  = []
for i ∈ 1:length(Xhist)
    Xi = SMACOF.align(Xhist[i], X)
    push!(lst_mse, SMACOF.mse(Xi, X))
    
    p1 = scatter(X[:, 1], X[:, 2], alpha=0.7, markersize=8, legend=nothing);
    scatter!(Xi[:,1], Xi[:, 2]);
    xlims!((-0.2, 1.2))
    ylims!((-0.2, 1.2))
    title!("$i")

    p2 = plot(1:i, lst_mse, legend=nothing, marker='o')
    xlims!((0, length(Xhist))) 
    ylims!((0, 0.35))  
    title!("MSE") 
    p = plot(p1, p2, layout=grid(1, 2, widths=[.6, .4]))
    plot!(size=(600, 400))
    frame(animation, p)
end
run(`ffmpeg -y -r 15 -i $abspath"%06d.png" -vcodec libx264 -crf 25 "$(gifdir)$(filename).mp4"`)
run(`ffmpeg -y -i $(gifdir)$(filename).mp4 -vf "fps=10,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $(gifdir)$(filename).gif`)
run(`rm $(gifdir)$(filename).mp4`)

# Clear image folder
for f in readdir(relpath)
    run(`rm $relpath/$(f)`)
end
```
<p align="center">
<img src="https://github.com/alainchau/SMACOF.jl/blob/main/examples/simple/simple.gif" width="600" height="400">
</p>
