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
Xhist, σ = wda_smacof(dists(X), η=0.8, verbose=true, anchors=X, ε=1e-6, return_history=true)[2:3]

# Make mp4 and then convert to gif
animation = Animation(abspath, String[])
lst_mse  = []
for i ∈ 1:length(Xhist)
    Xi = SMACOF.align(Xhist[i], X)
    push!(lst_mse, SMACOF.mse(Xi, X))
    
    p1 = scatter(X[:, 1], X[:, 2], alpha=0.7, markersize=8, label="true");
    scatter!(Xi[:,1], Xi[:, 2], label="smacof");
    xlims!((-0.2, 1.2))
    ylims!((-0.2, 1.2))
    title!("$i")

    p2 = plot(1:i, lst_mse, legend=nothing)
    xlims!((0, length(Xhist))) 
    ylims!((0, 0.35))  
    title!("MSE") 
    p = plot(p1, p2, layout=(1, 2))
    plot!(dpi=300,  size=(1000, 500))
    frame(animation, p)
end
run(`ffmpeg -y -r 15 -i $abspath"%06d.png" -vcodec libx264 -crf 25 "$(gifdir)$(filename).mp4"`)
run(`ffmpeg -y -i $(gifdir)$(filename).mp4 -vf "fps=10,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $(gifdir)$(filename).gif`)
run(`rm $(gifdir)$(filename).mp4`)

# Clear image folder
for f in readdir(relpath)
    run(`rm $relpath/$(f)`)
end