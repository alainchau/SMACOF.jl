using SMACOF
using Plots
gr()
ENV["GKSwstype"] = "100"

relpath = "examples/wda_smacof/gif_stems"
abspath = "/home/alain/code/SMACOF/examples/wda_smacof/gif_stems/"

function main(n=500)
    X = rand(2, n)
    
    X = X[:, .!((1 / 3 .< X[1,:] .< 2 / 3) .& (1 / 2 .< X[2,:]))]

    Y, Xhist = wda_smacof(dists(X), η=0.8, verbose=true, anchors=X, ε=1e-12, return_history=true)

    # Clear image folder
    for f in readdir(relpath)
        run(`rm $relpath/$(f)`)
    end

    # Make mp4 and then convert to gif
    animation = Animation(abspath, String[])
    for i ∈ 1:size(Xhist, 1)
        p = scatter(X[1,:], X[2,:], alpha=0.7, markersize=10, label="true");
        Xi = SMACOF.align(Xhist[i,:,:], X)
        scatter!(Xi[1,:], Xi[2,:], label="smacof");
        xlims!((-0.2, 1.2))
        ylims!((-0.2, 1.2))
        title!("$i")
        frame(animation, p)
    end
    run(`ffmpeg -y -r 15 -i $abspath"%06d.png" -vcodec libx264 -crf 25 "examples/wda_smacof/anim.mp4"`)
    run(`ffmpeg -y -i examples/wda_smacof/anim.mp4 -vf "fps=10,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 examples/wda_smacof/anim.gif`)
    run(`rm examples/wda_smacof/anim.mp4`)
end

main();
