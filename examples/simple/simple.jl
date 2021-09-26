using SMACOF
using Plots
using Random
Random.seed!(2021)
gr()

script_dir = dirname(@__FILE__)

X = rand(400, 2)
X = X[.!((1 / 4 .< X[:,1] .< 3 / 4) .& (1 / 4 .< X[:,2] .< 3 / 4)), :]
# Xhist = wda_smacof(dist(X), η = 0.8, verbose = true, anchors = X, ε = 1e-6, return_history = true)[2]
sm = smacof(dist(X), Xinit = randn(size(X)) * 0.1, anchors = X, return_hist = true)
Xhist = sm.hist
# Make mp4 and then convert to gif
anim_dir = joinpath(dirname(script_dir), "tmp")
animation = Animation(anim_dir, String[])
lst_mse  = []
lst_str = sm.σ / maximum(sm.σ) * 0.35
for i ∈ 1:length(Xhist)
    Xi = align(Xhist[i], X)
    push!(lst_mse, mse(Xi, X))

    p1 = scatter(X[:, 1], X[:, 2], alpha = 0.7, markersize = 8, legend = nothing);
    scatter!(Xi[:,1], Xi[:, 2]);
    xlims!((-0.2, 1.2))
    ylims!((-0.2, 1.2))
    title!("$i")

    p2 = plot(1:i, lst_mse, label = "mse", legend = :best)
    plot!(p2, 1:i, lst_str[1:i], label = "stress")
    xlims!((0, length(Xhist))) 
    ylims!((0, 0.35))  
    title!("Performance") 
    p = plot(p1, p2, layout = grid(1, 2, widths = [.6, .4]))
    plot!(size = (600, 400))
    frame(animation, p)
end

gif_path = joinpath(script_dir, "simple")
run(`ffmpeg -y -r 15 -i $anim_dir/"%06d.png" -vcodec libx264 -crf 25 "$(gif_path).mp4"`)
run(`ffmpeg -y -i $(gif_path).mp4 -vf "fps=10,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $(gif_path).gif`)
run(`rm $(gif_path).mp4`)

# Clear image folder
for f in readdir(anim_dir)
    println(f)
    run(`rm $(joinpath(anim_dir, f))`)
end