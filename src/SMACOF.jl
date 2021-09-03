module SMACOF

import Base:axes, getindex, setindex!, last, *, size
using LinearAlgebra
import LinearAlgebra:AbstractMatrix,diagind
using Distances
import LinearAlgebra:issymmetric, mul!
import Statistics: sum, mean

include("bmat.jl")
include("cmds.jl")
include("helpers.jl")
include("procrustes.jl")
include("smacof.jl")
include("wda_smacof.jl")

export classical_scaling
export distance_matrix
export mse
export stress, stress1
export smacof, SmacofParams
export wda_smacof

end