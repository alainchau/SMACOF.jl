using SMACOF
using Plots
using Random
gr()
ENV["GKSwstype"] = "100"
Random.seed!(2021)

include("../example_template.jl")

gifdir = "examples/wda_smacof/"
filename = "noiseless"

generate_gif((X, Δ) -> wda_smacof(Δ, 
                    η=0.9, 
                    Xinit=rand(size(X)...),
                    verbose=true, 
                    anchors=X, 
                    ε=1e-4, 
                    DA_itmax=30,
                    return_history=true)[2],
            gifdir, filename, n=500)
