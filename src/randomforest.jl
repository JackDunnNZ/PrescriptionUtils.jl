struct RandomForest
  forests::Vector
end

function train(d::Data{Normalized}, ::Val{:randomforest})
  M = length(d.y_T)

  forests = map(1:M) do m
    X_m = d.X_T[m]
    y_m = d.y_T[m]
    n_m = length(y_m)

    rf = ensemble.RandomForestRegressor(n_estimators=100)
    ScikitLearn.fit!(rf, X_m, y_m)

    rf
  end

  RandomForest(forests)
end

function getoutcomes(new_data::Data{Normalized}, rf::RandomForest)
  M = length(rf.forests)
  n = length(new_data.y)
  outcomes = Matrix{Float64}(undef, n, M)

  for m = 1:M
    outcomes[:, m] = ScikitLearn.predict(rf.forests[m], new_data.X)
  end

  outcomes
end
