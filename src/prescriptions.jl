function getoutcomes(train_data::Data{Normalized}, new_data::Data{Normalized},
                     method::Val)
  model = train(train_data, method)
  getoutcomes(new_data, model)
end
function getoutcomes(train_X, train_y, train_T, test_X, test_y, test_T,
                         method::Symbol)
  train_d = Data(train_X, train_y, train_T)
  test_d = Data(test_X, test_y, test_T)

  train_d_norm, μ, σ = normalize_data(train_d)
  test_d_norm = normalize_data(test_d, μ, σ)

  getoutcomes(train_d_norm, test_d_norm, Val{method}())
end

function makeprescriptions(
    outcomes::Matrix{Float64},
    allowed_prescriptions::Vector{Vector{Int}}=[collect(1:size(outcomes, 2))
                                                for _ in 1:size(outcomes, 1)],
  )
  map(1:size(outcomes, 1)) do i
    idx = indmin(outcomes[i, allowed_prescriptions[i]])
    allowed_prescriptions[i][idx]
  end
end

function predictoutcomes(outcomes::Matrix{Float64}, prescriptions::Vector{Int})
  map(1:length(prescriptions)) do i
    outcomes[i, prescriptions[i]]
  end
end
