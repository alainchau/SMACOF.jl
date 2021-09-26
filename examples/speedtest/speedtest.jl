using PyCall
using DataFrames
using SMACOF
using Plots
using Statistics
skl = pyimport("sklearn.manifold")

max_iter = 200
function skl_smacof(Δ; anchors = nothing)
    Yhat = skl.smacof(Δ, n_init = 1, eps = eps(), max_iter = max_iter)[1]
    return align(Yhat, anchors)
end

function main(nlst = [10,20, 30, 50, 100, 200, 300, 500], nrepeats = 10)
    first_run = true

    df = DataFrame(n = Int[], method = Symbol[], mse = Float64[], time = Float64[])
    for n in nlst
        for i in 1:nrepeats
            X = 2 * rand(n, 2)

            Δ = dist(X)
            t = time()
            Y_jl = smacof(Δ, anchors = X, ε = eps(), itmax = max_iter, Xinit = rand(n, 2))
            t_jl = time() - t
            # push!(df, [n, :julia, mse(Y, X), time() - t])

            t = time()
            Y_sk = skl_smacof(Δ, anchors = X)
            t_sk = time() - t
            # push!(df, [n, :sklearn, mse(Yhat, X), time() - t])

            if first_run
                first_run = false
                continue
            end            
            push!(df, [n, :julia, mse(Y_jl, X), t_jl])
            push!(df, [n, :sklearn, mse(Y_sk, X), t_sk])
        end
    end

    df_mean_times = combine(groupby(df, [:method, :n]), :time => mean => :mean)
    sort!(df_mean_times, :n)

    df_julia_times = df[df[!, :method] .== :julia, [:n, :time]]
    plot(legend = :outertopleft)
    scatter!(df_julia_times[!, :n], df_julia_times[!, :time], label = "julia", color = 1)
    plot!(df_mean_times[df_mean_times[!, :method] .== :julia, :n], df_mean_times[df_mean_times[!, :method] .== :julia, :mean], color = 1, label = nothing)

    df_sklearn_times = df[df[!, :method] .== :sklearn, [:n, :time]]
    scatter!(df_sklearn_times[!, :n], df_sklearn_times[!, :time], label = "sklearn", color = 2)
    plot!(df_mean_times[df_mean_times[!, :method] .== :sklearn, :n], df_mean_times[df_mean_times[!, :method] .== :sklearn, :mean], color = 2, label = nothing)

    title!("First to $max_iter iterations")
    xlabel!("sample size (n)")
    ylabel!("time (sec)")
    savefig(joinpath(dirname(@__FILE__), "results.png"))
    
end

main()