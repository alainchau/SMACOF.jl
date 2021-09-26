function wda_smacof(Δ; W = nothing, p = 2, X = nothing, anchors = nothing, ε = 1e-6, α = 0.9, Tmin = 1e-4, itmax = 100)
    to = TimerOutput()
    n = size(Δ, 1)

    # Generate random initial mapping
    @timeit to "Xinit" Xₖ = isnothing(X) ? rand(n, p) : X
    @timeit to "dist" Dₖ = dist(Xₖ)

    @timeit to "Winit" if isnothing(W)
        W = ones(n, n)
    end
    @timeit to "makeWidx" I = findall(W .< 1e-4)

    # Initialize T
    @timeit to "init Tk" Tₖ = 0.9 * maximum(Δ / sqrt(2 * p)) 
    Δₖ = zeros(size(Δ))
    
    V̇ = - Matrix(W)
    @timeit to "make V" begin
    for i in 1:n
        V̇[i, i] = -1.0
        V̇[i, i] = sum(view(V̇, :, i))
    end
    end
    @timeit to "Binit" B = zeros(n, n)

    # Storage
    @timeit to "initarrays" begin
    Xᵢ = rand(n, p)
    rᵢ = zeros(n, p)
    dᵢ = zeros(n, p)
    Vd = zeros(n, p)
    end
    k = 1
    @timeit to "mainloop" while Tₖ ≥ Tmin
        
        @timeit to "Δₖupdate" begin  for i in eachindex(Δ)
            i in I && continue
            Δₖ[i] = max(0, Δ[i] - Tₖ * sqrt(2p))
        end
        end

        σ₀ = 0.0
        @timeit to "stress" σ₁ = stress(Δₖ, Dₖ, W)
        
        @timeit to "mainiloop" while abs(σ₁ - σ₀) > ε
            # fill!(B, 0.0)
            @timeit to "updateB" begin
            for i in eachindex(B)
                B[i] = - W[i] * Δₖ[i] / (Dₖ[i] + eps())
            end
                for i in 1:n
                B[i, i] = 0.0
                B[i, i] = - sum(view(B, :, i))
                end
            end
            # Conjugate Gradient method
            # @timeit to "r" rᵢ[:] = B * Xₖ - V̇ * Xᵢ
            @timeit to "r" begin
                mul!(rᵢ, B, Xₖ)
                mul!(rᵢ, V̇, Xₖ, -1.0, 1.0)
            end
            @timeit to "d" dᵢ[:] = rᵢ
            @timeit to "CG" for i in 1:p
                @timeit to "β" βᵢ = 1 / dot(rᵢ, rᵢ)
                # @timeit to "α" begin αᵢ = 1 / dot(dᵢ, V̇, dᵢ)
                #     αᵢ /= βᵢ
                # end
                @timeit to "α" begin
                    mul!(Vd, V̇, dᵢ)
                    αᵢ = 1 / dot(dᵢ, Vd) / βᵢ
                end
                # @timeit to "Xi" Xᵢ += αᵢ * dᵢ
                @timeit to "Xi" axpy!(αᵢ, dᵢ, Xᵢ)
                # @timeit to "Vcache" mul!(Vcache, V̇, dᵢ)
                # @timeit to "r" rᵢ[:] -= αᵢ * Vcache
                @timeit to "r" mul!(rᵢ, V̇, dᵢ, αᵢ, 1.0)
                @timeit to "β" βᵢ *= dot(rᵢ, rᵢ)
                @timeit to "di" begin 
                    rmul!(dᵢ, βᵢ)
                axpy!(1.0, rᵢ, dᵢ)
                end
                end
            @timeit to "Xk=Xi" Xₖ = Xᵢ
                    # @timeit to "dist!" dist!(Dₖ, Xₖ)
                    @timeit to "dist!" pairwise!(Dₖ, Euclidean(), Xₖ, dims = 1)
            σ₀ = σ₁
                @timeit to "stressupdate" σ₁ = stress(Δₖ, Dₖ, W)
            println(σ₁)
        end
            
        Tₖ *= α
        k > itmax && break
        k += 1
    end
    X = align(Xₖ, anchors)
    show(to)
    return X
end            

# julia> @btime pairwise!($out, Euclidean(), $x, dims=1);
#   117.161 μs (2 allocations: 7.31 KiB)

# julia> @btime @tullio $out[i,j] = ($x[i,μ] - $x[j, μ])^2 |> sqrt;
#   169.234 μs (0 allocations: 0 bytes)