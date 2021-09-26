"""
    stress(Δ, Λ, W = ones(size(Δ)), iterable=eachindex(Δ))

Compute the stress: Σ wij (Δij -  Λij)^2
"""
function stress(Δ, Λ, W = ones(size(Δ)))
    n = size(Δ, 1)
    s = 0.0
    for idx in eachindex(Δ)
        s += W[idx] * (Δ[idx] - Λ[idx])^2
    end
    s    
end

function log_output(k, σ, verbose)
    verbose == false && return
    if k == 1
        println("Iteration\tStress")
        println("---------\t------")
    end
    @printf("%3d\t      %-6f\n", k, round(σ[k], digits = 6))
end