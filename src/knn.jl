struct KNN
  kcoeff::Float64
  neighbors::Data{Weighted}
  weights::Vector{Vector{Float64}}
end
get_bestk(n, knn::KNN) = round(Int, knn.kcoeff * sqrt(n))

"Find the best rule for k on the given data `d`."
function find_k_coeff(d::Data{Weighted}, k_range=1:100)
  M = length(d.y_T)

  best_k = zeros(Int, M)
  for m = 1:M
    X_m = d.X_T[m]
    y_m = d.y_T[m]
    n_m = length(y_m)

    max_k = min(maximum(k_range), n_m - 1)
    kdtree = NearestNeighbors.KDTree(Matrix(X_m'))
    # Find the k+1 nearest neighbors to each point, sorted by distance
    # Each point has itself as the first neighbor, which we will ignore
    idxs, _ = NearestNeighbors.knn(kdtree, Matrix(X_m'), max_k + 1, true)

    # Get the outcomes for each of the neighbors
    # Ignore the first neighbor since it's just the point itself
    cum_means = Vector{Vector{Float64}}(undef, n_m)
    for i = 1:n_m
      outcomes = y_m[idxs[i][2:end]]
      cum_means[i] = cumsum(outcomes) ./ (1:max_k)
    end

    # Find the k that minimizes the MAE of kNN predictions
    best_mae = Inf
    for k in k_range
      k >= n_m && continue
      mae = 0.0
      for i = 1:n_m
        mae += abs(cum_means[i][k] - y_m[i])
      end
      mae /= n_m

      if mae < best_mae
        best_mae = mae
        best_k[m] = k
      end
    end
  end

  sqrt_n = sqrt.(length.(d.y_T))
  path = GLMNet.glmnet(reshape(sqrt_n, M, 1), best_k)
  k_coeff = path.betas[1, end]
end

function getoutcomes(newdata_norm::Data{Normalized}, knn::KNN)
  M = length(knn.neighbors.X_T)

  n = size(newdata_norm.X, 1)
  outcomes = Matrix{Float64}(undef, n, M)

  for m = 1:M
    # Weight all points according to candidate treatment
    current_weighted = weight_data(newdata_norm, knn.weights, m)

    # Get all relevant weighted training points for candidate treatment
    X_m = knn.neighbors.X_T[m]
    y_m = knn.neighbors.y_T[m]

    k = get_bestk(length(y_m), knn)
    k = max(min(k, length(y_m)), 1)

    # Get k nearest neighbors for each new point from the neighbor pool
    # We don't need the neighbors in sorted order
    kdtree = NearestNeighbors.KDTree(Matrix(X_m'))
    idxs, _ = NearestNeighbors.knn(kdtree, Matrix(current_weighted.X'), k,
                                   false)

    # Get the mean outcome for each point from the neighbors
    for i = 1:n
      outcome = 0.0
      idx = idxs[i]
      for j = 1:k
        outcome += y_m[idx[j]]
      end
      outcomes[i, m] = outcome / k
    end
  end

  outcomes
end

"Trains kNN model on training data `d`."
function train(d::Data{Normalized}, ::Val{:knn})
  d_weighted, weights = weight_data(d)

  k_coeff = find_k_coeff(d_weighted)
  KNN(k_coeff, d_weighted, weights)
end

"Estimates counterfactuals on the dataset `d` using kNN."
function getcounterfactuals(d::Data{Normalized}, impute_method::Val{:knn})
  knn = train_knn(d, impute_method)
  getoutcomes(d, knn)
end
