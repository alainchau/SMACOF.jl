using SMACOF
using DelimitedFiles
using Distances
using Plots
using Shapefile
gr()

script_location = @__FILE__
script_dir = dirname(script_location)

dms2deg(d,m,s) = d + m / 60 + s / 3600

clld_path = joinpath(script_dir, "sgb128_lat_lon.txt")
cities_lat_lon_dms = readdlm(clld_path, skipstart = 7)
cities_lat_lon = zeros(size(cities_lat_lon_dms, 1), 2)
for i in 1:size(cities_lat_lon_dms, 1)
    lat = dms2deg(cities_lat_lon_dms[i, 1:3]...)
    lon = dms2deg(cities_lat_lon_dms[i, 5:7]...)
    cities_lat_lon[i, 1] = lat
    cities_lat_lon[i, 2] = -lon
end
cn_path = joinpath(script_dir, "sgb128_names.txt")
cities_names = readdlm(cn_path, ',', skipstart = 2)[:, 2]
cities_names = map(x -> x[2:end], cities_names)

Δ = dist(cities_lat_lon)
anchors = (idx = 1:128, pos = cities_lat_lon)

Xhist = smacof(Δ, Xinit = rand(size(Δ, 1), 2),  anchors = anchors, verbose = true, return_hist = true).hist
# Xhist = wda_smacof(Δ, η = 0.8, verbose = true, anchors = anchors, ε = 1e-6, return_history = true)[2]

# Make mp4 and then convert to gif
anim_dir = joinpath(dirname(script_dir), "tmp")
animation = Animation(anim_dir, String[])
# shp = Shapefile.shapes(Shapefile.Table("examples/cities/cb_2018_us_state_500k.shp"))
for i ∈ 1:length(Xhist)
    println(i)
    Xi = align(Xhist[i], anchors)
    p = scatter(Xi[:, 2], Xi[:, 1], alpha = 0.7, markersize = 3, legend = nothing);
    # plot!(shp)
    annotate!.(Xi[:,2], Xi[:,1], text.(cities_names, :black, :left, 7))
    xlims!((-130, -60))
    ylims!((20, 55))
    title!("$i")
    frame(animation, p)
end
gif_path = joinpath(script_dir, "cities")
run(`ffmpeg -y -r 15 -i $anim_dir/"%06d.png" -vcodec libx264 -crf 25 "$(gif_path).mp4"`)
run(`ffmpeg -y -i $(gif_path).mp4 -vf "fps=10,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $(gif_path).gif`)
run(`rm $(gif_path).mp4`)

# Clear image folder
for f in readdir(anim_dir)
    run(`rm $(joinpath(anim_dir, f))`)
end