module SMACOF

using Base:AbstractFloat
import Base:axes, getindex, setindex!, last, *
using LinearAlgebra
import LinearAlgebra:AbstractMatrix,diagind
using Distances
import LinearAlgebra:issymmetric, mul!
import Statistics: sum, mean

include("cmds.jl")
include("da_smacof.jl")
include("helpers.jl")
include("procrustes.jl")
include("smacof.jl")
include("wda_smacof.jl")

export classical_scaling
export da_smacof
export dists
export stress
export smacof
export wda_smacof

end