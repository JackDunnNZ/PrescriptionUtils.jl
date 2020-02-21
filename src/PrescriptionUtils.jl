module PrescriptionUtils

using Compat

import GLMNet
import MLDataUtils
import NearestNeighbors
import PyCall
import ScikitLearn
import StatsBase

import Compat.Statistics: mean

export
  evaluatebaseline,
  evaluateoracle,
  evaluateprescriptions,
  getcounterfactuals,
  getoutcomes,
  makeprescriptions,
  predictoutcomes

# @sk_import creates a global pointer that doesn't work with precompilation
# copy directly: https://github.com/cstjean/ScikitLearn.jl/issues/50
const ensemble = PyCall.PyNULL()

function __init__()
    copy!(ensemble, PyCall.pyimport("sklearn.ensemble"))
end

include("data.jl")
include("prescriptions.jl")
include("evaluation.jl")

include("knn.jl")
include("lasso.jl")
include("randomforest.jl")


end # module
