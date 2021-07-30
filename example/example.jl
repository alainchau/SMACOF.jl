using SMACOF
using Plots
using Random
gr()
ENV["GKSwstype"] = "100"

function main(n=500)
    X = rand(2, n)
    Xinit = rand(2, n)
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
        frame(animation, p)
    end
    run(`ffmpeg -y -r 15 -i $loadpath"%06d.png" -vcodec libx264 -crf 25 "anim.mp4"`)
    run(`ffmpeg -y -i anim.mp4 -vf "fps=10,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 anim.gif`)
    run(`rm anim.mp4`)

end

main();
