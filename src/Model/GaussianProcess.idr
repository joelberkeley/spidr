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
||| This module contains the `GaussianProcess` type for defining a Gaussian process, along with
||| functionality for training and inference.
module Model.GaussianProcess

import Data.Nat
import Tensor
import Model.Kernel
import Model.MeanFunction
import Optimize
import Distribution
import Util

||| A Gaussian process is a collection of random variables, any finite number of which have joint
||| Gaussian distribution. It can be viewed as a function from a feature space to a joint Gaussian
||| distribution over a target space.
public export
data GaussianProcess : (0 features : Shape) -> Type where
  ||| Construct a `GaussianProcess` as a pair of mean function and kernel.
  MkGP : MeanFunction features -> Kernel features -> GaussianProcess features

||| The marginal distribution of the Gaussian process at the specified feature values.
|||
||| @at The feature values at which to evaluate the marginal distribution.
export
marginalise : {s : Nat}
  -> GaussianProcess features
  -> Tensor ((S s) :: features) Double
  -> Gaussian [] (S s)
marginalise (MkGP mean_function kernel) x = MkGaussian (mean_function x) (kernel x x)

posterior : {s : Nat}
 -> (prior : GaussianProcess features)
 -> (likelihood : Gaussian [] (S s))
 -> (training_data : (Tensor ((S s) :: features) Double, Tensor [S s] Double))
 -> GaussianProcess features
posterior (MkGP prior_meanf prior_kernel) (MkGaussian _ cov) (x_train, y_train) =
  let l = cholesky (prior_kernel x_train x_train + cov)
      alpha = l.T \\ (l \\ y_train)

      posterior_meanf : MeanFunction features
      posterior_meanf x = prior_meanf x + (prior_kernel x x_train) @@ alpha

      posterior_kernel : Kernel features
      posterior_kernel x x' = prior_kernel x x' -
                              (l \\ (prior_kernel x_train x)).T @@ (l \\ (prior_kernel x_train x'))

   in MkGP posterior_meanf posterior_kernel

log_marginal_likelihood : {s : Nat}
 -> GaussianProcess features
 -> Gaussian [] (S s)
 -> (Tensor ((S s) :: features) Double, Tensor [S s] Double)
 -> Tensor [] Double
log_marginal_likelihood (MkGP _ kernel) (MkGaussian _ cov) (x, y) =
  let l = cholesky (kernel x x + cov)
      alpha = l.T \\ (l \\ y)
      n = const {shape=[]} $ cast (S s)
      log2pi = log $ const {shape=[]} $ 2.0 * PI
      two = const {shape=[]} 2
   in - y @@ alpha / two - trace (log l) - n * log2pi / two

||| Find the hyperparameters which maximize the marginal likelihood for the specified structures of
||| prior and likelihood, then return the posterior for that given prior, likelihood and
||| hyperparameters.
|||
||| @optimizer The optimization tactic.
||| @mk_prior Constructs the prior from *all* the hyperparameters.
||| @mk_likelihood Constructs the likelihood from *all* the hyperparameters.
||| @training_data The observed data.
export
fit : {s : Nat}
 -> (optimizer: Optimizer hp)
 -> (mk_prior : hp -> GaussianProcess features)
 -> (mk_likelihood : hp -> Gaussian [] (S s))
 -> (training_data : (Tensor ((S s) :: features) Double, Tensor [S s] Double))
 -> GaussianProcess features
fit optimizer mk_prior mk_likelihood training_data =
  let objective : hp -> Tensor [] Double
      objective hp = log_marginal_likelihood (mk_prior hp) (mk_likelihood hp) training_data

      params := optimizer objective

   in posterior (mk_prior params) (mk_likelihood params) training_data
