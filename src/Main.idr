import Tensor

import Data.Vect
import GaussianProcess
import Kernel
import MeanFunction
--import Optimize

-- note having to specify types isn't much of a problem, because they'll usually
--   be inferred from type signatures
two_by_four : Tensor [2, 4] Double
two_by_four = MkTensor [[-1, -2, -3, -4], [1, 2, 3, 4]]

four_by_three : Tensor [4, 3] Double
four_by_three = MkTensor [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

one_by_one_by_two : Tensor [1, 1, 2] Double
one_by_one_by_two = MkTensor [[[2, 1]]]

three_by_one_by_two : Tensor [3, 1, 2] Double
three_by_one_by_two = MkTensor [[[1, 2]], [[3, 4]], [[5, 6]]]

three_by_three : Tensor [3, 3] Double
three_by_three = MkTensor [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

div : Tensor [1, 1, 3] Double
-- div = (the (Tensor [1, 1, 3] Double) (MkTensor [[[1, 2, 3]]])) / ?a
div = (the (Tensor [1, 1, 3] _) (MkTensor [[[1, 2, 3]]])) / (the (Tensor [1, 1] _) (MkTensor [[4]]))

main : IO ()
main = let MkGaussian mean cov = marginalise (MkGP zero linear) two_by_four in do printLn (mean, cov)

{--
mk_gp : Tensor [] Double -> GaussianProcess [1]
mk_gp variance = MkGP zero $ radial_basis_function variance

training_data : (Tensor [25, 1] Double, Tensor [25] Double)

optimizer : Optimizer $ Tensor [] Double
optimizer = grid_search (MkTensor [100]) (MkTensor [-1]) (MkTensor [1])

optimized_gp : GaussianProcess [1]
optimized_gp = mk_gp $ optimize optimizer mk_gp training_data

posterior : Maybe $ GaussianProcess [1]
posterior = let likelihood = MkGaussian (MkTensor $ replicate 25 0) (diag 25 1) in
  gp_bayes optimized_gp likelihood training_data
--}
