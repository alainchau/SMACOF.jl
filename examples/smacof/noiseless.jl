using SMACOF
using Plots
using Random
gr()
ENV["GKSwstype"] = "100"
Random.seed!(2021)

include("../example_template.jl")

gifdir = "examples/smacof/"
filename = "noiseless"

function f(X, Δ)
    sm = Smacof(Δ, Xinit=rand(size(X)...), verbose=true, itmax=100)
    fit(sm)
    return sm.Xhist
end

generate_gif(f, gifdir, filename, n=100)