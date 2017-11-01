"Different states the data can be in."
abstract type DataState end

"Starting state for data."
immutable Original <: DataState end
"State for data with normalized `X` values."
immutable Normalized <: DataState end
"State for data with normalized and weighted `X` values."
immutable Weighted <: DataState end

"""
Data container for observational data.

Contains `X` matrix of covariates, `y` vector of outcomes, and `T` vector of
treatments.

Also provides `X_T` and `y_T` containing the subset of the data for each
treatment.
"""
immutable Data{DS <: DataState}
  X::Matrix{Float64}
  y::Vector{Float64}
  T::Vector{Int}
  X_T::Vector{Matrix{Float64}}
  y_T::Vector{Vector{Float64}}

  function Data{DS}(X, y, T) where DS
    treatment_inds = [find(T .== t) for t in 1:maximum(T)]
    X_T = [X[inds, :] for inds in treatment_inds]
    y_T = [y[inds] for inds in treatment_inds]
    new(X, y, T, X_T, y_T)
  end
end

# Set controls on how data containers can be created
function Data(X, y::AbstractVector{Float64}, T::AbstractVector{Int})
  Data{Original}(
      convert(Matrix{Float64}, X),
      convert(Vector{Float64}, y),
      convert(Vector{Int}, T),
  )
end
function Data{DS}(d::Data{Original}, X_norm) where DS <: Normalized
  Data{Normalized}(X_norm, d.y, d.T)
end
function Data{DS}(d::Data{Normalized}, X_norm) where DS <: Weighted
  Data{Weighted}(X_norm, d.y, d.T)
end

"Normalize data `d` so that it is centered and scaled."
function normalize_data(d::Data{Original})
  X_norm = copy(d.X)
  μ, σ = MLDataUtils.rescale!(X_norm, MLDataUtils.ObsDim.First())
  d_norm = Data{Normalized}(d, X_norm)
  d_norm, μ, σ::Vector{Float64}
end
"Normalize data `d` following transform of `μ` and `σ`."
function normalize_data(d::Data{Original}, μ::Vector{Float64},
                        σ::Vector{Float64})
  X_norm = copy(d.X)
  MLDataUtils.rescale!(X_norm, μ, σ, MLDataUtils.ObsDim.First())
  Data{Normalized}(d, X_norm)
end

"Weight data `d` according to linear regression weights fitted to data."
function weight_data(d::Data{Normalized})
  weights = map(1:length(d.X_T)) do m
    X_m = d.X_T[m]
    y_m = d.y_T[m]
    model = ScikitLearn.fit!(LinearRegression(), X_m, y_m)
    coef = model[:coef_]::Vector{Float64}
    w = abs.(coef)
    w ./= sum(w)
  end

  # Apply weights
  d_weighted = weight_data(d, weights)

  d_weighted, weights
end
"Weight data `d` according to `weights` for the treatments in `d.T`."
function weight_data(d::Data{Normalized}, weights::Vector{Vector{Float64}})
  X_weighted = copy(d.X)
  for i = 1:size(X_weighted, 1)
    X_weighted[i, :] .*= weights[d.T[i]]
  end
  Data{Weighted}(d, X_weighted)
end
"Weight data `d` according to `weights` for a single `treatment`."
function weight_data(d::Data{Normalized}, weights::Vector{Vector{Float64}},
                     treatment::Int)
  X_weighted = copy(d.X)
  X_weighted .*= weights[treatment]'
  Data{Weighted}(d, X_weighted)
end
