immutable Lasso
  models::Vector{Tuple{Vector{Float64},Float64}}
end

function train(d::Data{Normalized}, ::Val{:lasso})
  M = length(d.y_T)

  models = map(1:M) do m
    X_m = d.X_T[m]
    y_m = d.y_T[m]
    n_m = length(y_m)

    sn = round(Int, n_m * 0.75)
    tr_inds = StatsBase.sample(1:n_m, sn, replace=false)
    vl_inds = setdiff(1:n_m, tr_inds)

    tr_X = X_m[tr_inds, :]
    vl_X = X_m[vl_inds, :]
    tr_y = y_m[tr_inds]
    vl_y = y_m[vl_inds]

    if n_m < 10
      zeros(size(X_m, 2)), mean(y_m)
    else
      cv = GLMNet.glmnetcv(tr_X, tr_y)
      best = indmin(cv.meanloss)
      β = full(cv.path.betas[:, best])
      β0 = cv.path.a0[best]
      β, β0
    end
  end

  Lasso(models)
end

function getoutcomes(new_data::Data{Normalized}, lasso::Lasso)
  M = length(lasso.models)
  n = length(new_data.y)
  outcomes = Matrix{Float64}(n, M)

  for m = 1:M
    β, β0 = lasso.models[m]
    outcomes[:, m] = new_data.X * β + β0
  end

  outcomes
end


