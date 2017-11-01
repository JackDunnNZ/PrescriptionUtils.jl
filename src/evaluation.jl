"Calculates counterfactuals for `d` using `impute_method`."
function getcounterfactuals(d::Data{Normalized}, impute_method::Symbol)
  cf_model = train(d, Val{impute_method}())
  cf = getoutcomes(d, cf_model)

  # Add in current prescriptions where known
  for i = 1:size(d.X, 1)
    cf[i, d.T[i]] = d.y[i]
  end

  cf
end
function getcounterfactuals(d::Data{Original}, impute_method::Symbol)
  d_norm, _, _ = normalize_data(d)
  getcounterfactuals(d_norm, impute_method)
end
function getcounterfactuals(X, y, T, impute_method::Symbol)
  getcounterfactuals(Data(X, y, T), impute_method)
end

"""
Evaluates quality of omniprescient oracle on data `d` subject to
`allowed_prescriptions`.
"""
function evaluateoracle(
    cf::Matrix{Float64},
    allowed_prescriptions::Vector{Vector{Int}}=[collect(1:size(cf, 2))
                                                for _ in 1:size(cf, 1)],
  )
  # For each sample, find the best allowed prescription
  n = length(allowed_prescriptions)
  outcomes = Vector{Float64}(n)
  prescriptions = Vector{Int}(n)
  for i = 1:n
    outcomes[i], prescriptions[i] = findmin(cf[i, allowed_prescriptions[i]])
  end

  outcomes, prescriptions
end

"Evaluates quality of `prescriptions` on data `d`."
function evaluateprescriptions(cf::Matrix{Float64}, prescriptions::Vector{Int})
  # For each sample, find result under the current prescription
  n = length(prescriptions)
  outcomes = Vector{Float64}(n)
  for i = 1:n
    outcomes[i] = cf[i, prescriptions[i]]
  end
  outcomes
end

"Evaluates quality of baseline standard-of-care model on data `d`."
function evaluatebaseline(cf::Matrix{Float64}, d::Data)
  evaluateprescriptions(cf, d.T)
end
evaluatebaseline(cf, X, y, T) = evaluatebaseline(cf, Data(X, y, T))
