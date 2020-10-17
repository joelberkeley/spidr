module GaussianProcess

import Tensor
import Data.Vect
import Kernel
import MeanFunction

-- todo work out whether we should parametrize over samples. It does feel suspicious
record GaussianProcess (samples : Nat) (features : Shape {rank}) where
  constructor MkGP
  mean_function : MeanFunction samples features
  kernel : Kernel samples features

GP : Nat -> Shape -> Type
GP = GaussianProcess

-- marginal_likelihood : Distribution ty => ty -> GaussianProcess -> GaussianProcess
-- marginal_likelihood

record Gaussian (d : Nat) where
  constructor MkGaussian
  mean : Tensor [d] Double
  cov : Tensor [d, d] Double

Likelihood : Type

bayes_rule : GP samples features -> Likelihood -> GP samples features
bayes_rule prior likelihood = ?posterior

-- of note: since we can't do matrix multiplication with empty vectors, we can't do bayes without any training data
gpr_bayes : (prior : GP (S samples) features) -> (likelihood : Gaussian (S samples)) -> (training_data: (Tensor ((S samples) :: features) Double, Tensor [S samples] Double)) -> Maybe $ GP (S samples) features
gpr_bayes {features} {samples} (MkGP mean_function kernel) (MkGaussian mean cov) (x_train, y_train) with (inverse (kernel x_train x_train + cov))
  | Just inv = Just $ MkGP posterior_mean_function posterior_kernel where
    posterior_mean_function : MeanFunction (S samples) features
    posterior_mean_function x = (kernel x x_train) @@ inv @@ y_train

    posterior_kernel : Kernel (S samples) features
    posterior_kernel x x' = kernel x x' - (kernel x x_train) @@ inv @@ (kernel x_train x')
  | Nothing = Nothing

minimize : (a -> Double) -> a

optimize : GP samples features -> (Tensor (samples :: features) dtype, Tensor [samples] dtype) -> GP samples features
optimize gp (x, y) = ?optimized_gp

-- is "marginalise" a better name for this function?
predict : GP samples features -> Tensor (samples :: features) Double -> Gaussian samples
predict (MkGP mean_function kernel) x = MkGaussian (mean_function x) (kernel x x)
