using SMACOF
using Plots
using Random
gr()
ENV["GKSwstype"] = "100"

relpath = "examples/smacof/gif_stems"
abspath = "/home/alain/code/SMACOF/examples/smacof/gif_stems/"
gifdir = "examples/smacof/"
filename = "noisy"

function add_noise_to_dist(D; noise=0.20)
    for i in 1:n
        for j in (i + 1):n
            # ! try: (1 + V) / (2 + V), V ~ Exp(λ) can scale λ
            k = (randexp() - 1) * noise + 1
            D[i, j] = k * D[i, j]
            D[j, i] = k * D[j, i]
        end
    end
    return D
end

function main(n=500)
    X = 2 * rand(2, n)
    D = add_noise_to_dist(dists(X), ones(Bool, n, n); noise=0.20)
    sm = Smacof(D, Xinit=rand(2, n), verbose=true, itmax=100)
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
        xlims!((-0.2, 2.2))
        ylims!((-0.2, 2.2))
        title!("$i")
        frame(animation, p)
    end
    run(`ffmpeg -y -r 15 -i $abspath"%06d.png" -vcodec libx264 -crf 25 "$(gifdir)$(filename).mp4"`)
    run(`ffmpeg -y -i $(gifdir)$(filename).mp4 -vf "fps=10,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $(gifdir)$(filename).gif`)
    run(`rm $(gifdir)$(filename).mp4`)
end

main();
