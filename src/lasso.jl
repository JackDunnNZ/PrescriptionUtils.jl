struct Lasso
  models::Vector{Tuple{Vector{Float64},Float64}}
end

function train(d::Data{Normalized}, ::Val{:lasso})
  M = length(d.y_T)

  models = map(1:M) do m
    X_m = d.X_T[m]
    y_m = d.y_T[m]
    n_m = length(y_m)

    if n_m < 10
      zeros(size(X_m, 2)), mean(y_m)
    else
      cv = GLMNet.glmnetcv(X_m, y_m)
      best = argmin(cv.meanloss)
      β = Vector(cv.path.betas[:, best])
      β0 = cv.path.a0[best]
      β, β0
    end
  end

  Lasso(models)
end

function getoutcomes(new_data::Data{Normalized}, lasso::Lasso)
  M = length(lasso.models)
  n = length(new_data.y)
  outcomes = Matrix{Float64}(undef, n, M)

  for m = 1:M
    β, β0 = lasso.models[m]
    outcomes[:, m] = new_data.X * β .+ β0
  end

  outcomes
end


