function wda_smacof(Δ; W = nothing, p = 2, X = nothing, anchors = nothing, ε = 1e-6, α = 0.9, Tmin = 1e-6, itmax = 100)
    n = size(Δ, 1)

    # Generate random initial mapping
    Xₖ = isnothing(X) ? rand(n, p) : X
    Dₖ = dist(Xₖ)

    if isnothing(W)
        W = ones(n, n)
    end
    I = findall(W .< 1e-4)

    # Initialize T
    Tₖ = (1 - eps()) *  maximum(Δ / sqrt(2 * p)) 
    Δₖ = zeros(size(Δ))
    
    V̇ = - Matrix(W)
    for i in 1:n
        V̇[i, i] = -1.0
        V̇[i, i] = sum(view(V̇, :, i))
    end
    B = zeros(n, n)

    # Storage
    Xᵢ = rand(n, p)
    rᵢ = zeros(n, p)
    dᵢ = zeros(n, p)
    Vd = zeros(n, p)
    k = 1
    while true
        if Tₖ < Tmin 
            break
        end
        for i in eachindex(Δ)
            i in I && continue
            Δₖ[i] = max(0, Δ[i] - Tₖ * sqrt(2p))
        end

        σ₀ = 0.0
        σ₁ = stress(Δₖ, Dₖ, W)
        
        while abs(σ₁ - σ₀) > ε
            # fill!(B, 0.0)
            for i in eachindex(B)
                B[i] = - W[i] * Δₖ[i] / (Dₖ[i] + eps())
            end
            for i in 1:n
                B[i, i] = 0.0
                B[i, i] = - sum(view(B, :, i))
            end
            # Conjugate Gradient method
            # @timeit to "r" rᵢ[:] = B * Xₖ - V̇ * Xᵢ
            mul!(rᵢ, B, Xₖ)
            mul!(rᵢ, V̇, Xₖ, -1.0, 1.0)

            dᵢ[:] = rᵢ
            for i in 1:p
                βᵢ = 1 / dot(rᵢ, rᵢ)
                mul!(Vd, V̇, dᵢ)
                αᵢ = 1 / dot(dᵢ, Vd) / βᵢ
                axpy!(αᵢ, dᵢ, Xᵢ)
                mul!(rᵢ, V̇, dᵢ, αᵢ, 1.0)
                βᵢ *= dot(rᵢ, rᵢ)
                rmul!(dᵢ, βᵢ)
                axpy!(1.0, rᵢ, dᵢ)
                end
            copyto!(Xₖ, Xᵢ)
            dist!(Dₖ, Xₖ)
            σ₀ = σ₁
            σ₁ = stress(Δₖ, Dₖ, W)
        end
            
        Tₖ *= α
        k > itmax && break
        k += 1
        println(k, "\t", σ₁)
    end
    smacof(Δ, Xinit = Xₖ, anchors = anchors)
end