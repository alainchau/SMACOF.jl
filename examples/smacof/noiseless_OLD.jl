using SMACOF
using Plots
gr()
ENV["GKSwstype"] = "100"

relpath = "examples/smacof/gif_stems"
abspath = "/home/alain/code/SMACOF/examples/smacof/gif_stems/"
gifdir = "examples/smacof/"
filename = "noiseless"

function main(n=500)
    X = rand(2, n)
    X = X[:, .!((1 / 3 .< X[1,:] .< 2 / 3) .& (1 / 2 .< X[2,:]))]
    n = size(X, 2)
    Xinit = rand(2, n)
    sm = Smacof(dists(X), Xinit=Xinit, verbose=true, itmax=100)
    fit(sm)

    # Clear image folder
    for f in readdir(relpath)
        run(`rm $relpath/$(f)`)
    end

    # Make mp4 and then convert to gif
    animation = Animation(abspath, String[])
    for i ∈ 1:sm.it[1]
        p = scatter(X[1,:], X[2,:], alpha=0.7, markersize=10, label="true");
        Xi = SMACOF.align(sm.Xhist[i,:,:], X)
        scatter!(Xi[1,:], Xi[2,:], label="smacof");
        xlims!((-0.2, 1.2))
        ylims!((-0.2, 1.2))
        title!("$i")
        frame(animation, p)
    end
    run(`ffmpeg -y -r 15 -i $abspath"%06d.png" -vcodec libx264 -crf 25 "$(gifdir)$(filename).mp4"`)
    run(`ffmpeg -y -i $(gifdir)$(filename).mp4 -vf "fps=10,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $(gifdir)$(filename).gif`)
    run(`rm $(gifdir)$(filename).mp4`)
end

main();
