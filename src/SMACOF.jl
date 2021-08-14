module SMACOF

using Distances
using LinearAlgebra
import LinearAlgebra.issymmetric
# import MultivariateStats: transform, fit
# using MultivariateStats
import Statistics.mean

include("cmds.jl")
include("da_smacof.jl")
include("helpers.jl")
include("procrustes.jl")
include("smacof.jl")
include("wda_smacof.jl")

# export classical_mds
export classical_scaling
export da_smacof
export distortion
export dists
export fit
export gethist
export stress
export Smacof
export wda_smacof

end