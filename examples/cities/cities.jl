using SMACOF
using DelimitedFiles
using Distances
using Plots
using Shapefile
gr()
ENV["GKSwstype"] = "100"

relpath = "examples/tmp/"
pathabs = "/home/alain/code/SMACOF/examples/tmp/"
gifdir = "examples/cities/"
filename = "cities"

dms2deg(d,m,s) = d + m / 60 + s / 3600

cities_lat_lon_dms = readdlm("examples/cities/sgb128_lat_lon.txt", skipstart=7)
cities_lat_lon = zeros(size(cities_lat_lon_dms, 1), 2)
for i in 1:size(cities_lat_lon_dms, 1)
    lat = dms2deg(cities_lat_lon_dms[i, 1:3]...)
    lon = dms2deg(cities_lat_lon_dms[i, 5:7]...)
    cities_lat_lon[i, 1] = lat
    cities_lat_lon[i, 2] = -lon
end
cities_names = readdlm("examples/cities/sgb128_names.txt", ',', skipstart=2)[:, 2]
cities_names = map(x -> x[2:end], cities_names)

Δ = dists(cities_lat_lon)
anchors = (idx = 1:128, pos = cities_lat_lon)

Xhist = wda_smacof(Δ, η=0.8, verbose=true, anchors=anchors, ε=1e-6, return_history=true)[2]

# Make mp4 and then convert to gif
animation = Animation(pathabs, String[])
# shp = Shapefile.shapes(Shapefile.Table("examples/cities/cb_2018_us_state_500k.shp"))
for i ∈ 1:length(Xhist)
    println(i)
    Xi = SMACOF.align(Xhist[i], anchors)
    p = scatter(Xi[:, 2], Xi[:, 1], alpha=0.7, markersize=3, legend=nothing);
    # plot!(shp)
    annotate!.(Xi[:,2], Xi[:,1], text.(cities_names, :black, :left, 7))
    xlims!((-130, -60))
    ylims!((20, 55))
    title!("$i")
    frame(animation, p)
end
run(`ffmpeg -y -r 15 -i $pathabs"%06d.png" -vcodec libx264 -crf 25 "$(gifdir)$(filename).mp4"`)
run(`ffmpeg -y -i $(gifdir)$(filename).mp4 -vf "fps=10,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $(gifdir)$(filename).gif`)
run(`rm $(gifdir)$(filename).mp4`)

# Clear image folder
for f in readdir(relpath)
    run(`rm $relpath/$(f)`)
end