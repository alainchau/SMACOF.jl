module SMACOF

using LinearAlgebra:Matrix
using Reexport
@reexport using DSTools
using LinearAlgebra
using Distances
using Printf

import Base:axes, getindex, setindex!, last, *, size
import LinearAlgebra:AbstractMatrix,diagind
import LinearAlgebra:issymmetric, mul!
import Statistics: sum, mean

include("cmds.jl")
include("helpers.jl")
include("procrustes.jl")
include("smacof.jl")
include("wda_smacof.jl")

export align, align!
export classical_scaling
export stress
export smacof
export wda_smacof

end