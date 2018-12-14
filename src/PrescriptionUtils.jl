module PrescriptionUtils

using Compat

import GLMNet
import MLDataUtils
import NearestNeighbors
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

ScikitLearn.@sk_import ensemble: RandomForestRegressor
ScikitLearn.@sk_import linear_model: LinearRegression

include("data.jl")
include("prescriptions.jl")
include("evaluation.jl")

include("knn.jl")
include("lasso.jl")
include("randomforest.jl")


end # module
