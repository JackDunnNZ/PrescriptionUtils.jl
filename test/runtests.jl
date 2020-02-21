using PrescriptionUtils
using Test


n = 100
p = 5
X = rand(n, p)
y = rand(n)
T = rand(1:3, n)

@testset "Methods" for method in (:knn, :lasso, :randomforest)
  getcounterfactuals(X, y, T, method)
end
