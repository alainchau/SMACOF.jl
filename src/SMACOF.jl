module SMACOF

using Distances
using LinearAlgebra
import MultivariateStats: transform, fit
using MultivariateStats
using Preconditioners
import Statistics.mean

include("da_smacof.jl")
include("helpers.jl")
include("procrustes.jl")
include("smacof.jl")
include("wda_smacof.jl")

export classical_mds
export da_smacof
export distortion
export dists
export fit
export gethist
export stress
export Smacof
export wda_smacof

end