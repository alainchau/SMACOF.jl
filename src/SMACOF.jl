module SMACOF

using Distances
using LinearAlgebra
import MultivariateStats: transform, fit
using MultivariateStats
import Statistics.mean
include("helpers.jl")
include("procrustes.jl")
include("smacof.jl")

export Smacof, fit, stress, distortion, classical_mds, gethist, dists


end