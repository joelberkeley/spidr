{--
Copyright 2021 Joel Berkeley

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--}
||| This module contains functionality for Gaussian process inference.
module Model.GaussianProcess

import Tensor
import Model
import Model.Kernel
import Model.MeanFunction
import Optimize
import Distribution
import Util

||| A Gaussian process is a collection of random variables, any finite number of which have joint
||| Gaussian distribution. It can be viewed as a function from a feature space to a joint Gaussian
||| distribution over a target space.
|||
||| @features The shape of the feature domain.
public export
data GaussianProcess : (0 features : Shape) -> Type where
  ||| Construct a `GaussianProcess` as a pair of mean function and kernel.
  MkGP : MeanFunction features -> Kernel features -> GaussianProcess features

posterior :
  GaussianProcess features
  -> Tensor [] Double
  -> forall s . (Tensor ((S s) :: features) Double, Tensor [S s] Double)
  -> GaussianProcess features
posterior (MkGP prior_meanf prior_kernel) noise (x_train, y_train) =
  let l = cholesky (prior_kernel x_train x_train + diag {n=S s} noise)
      alpha = l.T \\ (l \\ y_train)

      posterior_meanf : MeanFunction features
      posterior_meanf x = prior_meanf x + (prior_kernel x x_train) @@ alpha

      posterior_kernel : Kernel features
      posterior_kernel x x' = prior_kernel x x' -
                              (l \\ (prior_kernel x_train x)).T @@ (l \\ (prior_kernel x_train x'))

   in MkGP posterior_meanf posterior_kernel

log_marginal_likelihood :
  GaussianProcess features
  -> Tensor [] Double
  -> {s : _} -> (Tensor ((S s) :: features) Double, Tensor [S s] Double)
  -> Tensor [] Double
log_marginal_likelihood (MkGP _ kernel) noise (x, y) =
  let l = cholesky (kernel x x + diag {n=S s} noise)
      alpha = l.T \\ (l \\ y)
      n = const {shape=[]} $ cast (S s)
      log2pi = log $ const {shape=[]} $ 2.0 * PI
      two = const {shape=[]} 2
   in - y @@ alpha / two - trace (log l) - n * log2pi / two

||| A trainable model implementing vanilla Gaussian process regression. That is, regression with a
||| Gaussian process as conjugate prior for homoscedastic Gaussian likelihoods. See the following
||| for details:
|||
||| Gaussian Processes for Machine Learning
||| Carl Edward Rasmussen and Christopher K. I. Williams
||| The MIT Press, 2006. ISBN 0-262-18253-X.
|||
||| or
|||
||| Pattern Recognition and Machine Learning, Christopher M. Bishop
public export
data ConjugateGPRegression : (0 features : Shape) -> Type where
  MkConjugateGPR : (Tensor [p] Double -> GaussianProcess features) -> Tensor [p] Double
                   -> Tensor [] Double -> ConjugateGPRegression features

||| Construct a probabilistic model for the latent target values.
export
predict_latent : ConjugateGPRegression features
  -> ProbabilisticModel features {marginal=Gaussian [1]}
predict_latent (MkConjugateGPR mk_gp gp_params _) x =
  let (MkGP meanf kernel) = mk_gp gp_params
    in MkGaussian (expand 1 $ meanf x) (expand 2 $ kernel x x)

||| Fit the Gaussian process and noise to the specified data.
export
fit : ConjugateGPRegression features
  -> (forall n . Tensor [n] Double -> Optimizer $ Tensor [n] Double)
  -> {s : _} -> Data {samples=S s} features [1]
  -> ConjugateGPRegression features
fit (MkConjugateGPR {p} mk_prior gp_params noise) optimizer {s} training_data =
  let objective : Tensor [S p] Double -> Tensor [] Double
      objective params = let (noise, prior_params) = split 1 params
                             (x, y) = training_data
                          in log_marginal_likelihood (mk_prior prior_params)
                             (squeeze noise) (x, squeeze y)

      (noise, gp_params) := split 1 $ optimizer (concat (expand 0 noise) gp_params) objective

      mk_posterior : Tensor [p] Double -> GaussianProcess features
      mk_posterior params' = let (x, y) = training_data
                              in posterior (mk_prior params') (squeeze noise) (x, squeeze y)

    in MkConjugateGPR mk_posterior gp_params (squeeze noise)
