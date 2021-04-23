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
module GaussianProcess

import Tensor
import Data.Vect
import Data.Nat
import Kernel
import MeanFunction
import Optimize
import Distribution

public export
data GaussianProcess : (0 features : Shape) -> Type where
  MkGP : MeanFunction features -> Kernel features -> GaussianProcess features

-- todo implement for no training data
-- todo we don't use the likelihood mean. Is that right?
export
posterior : {s : Nat}
 -> (prior : GaussianProcess features)
 -> (likelihood : Gaussian (S s) [])
 -> (training_data : (Tensor ((S s) :: features) Double, Tensor [S s] Double))
 -> Maybe $ GaussianProcess features
posterior (MkGP mean_function kernel) (MkGaussian _ cov) (x_train, y_train) = map foo $ inverse (kernel x_train x_train + cov) where
  foo : Tensor [S s, S s] Double -> GaussianProcess features
  foo inv = MkGP posterior_mean_function posterior_kernel where
    posterior_mean_function : MeanFunction features
    -- todo can we use rewrite to avoid the use of implicits here and for posterior_kernel?
    posterior_mean_function {sm} x = mean_function x + (@@) {head=[_]} ((@@) {head=[_]} (kernel x x_train) inv) y_train

    posterior_kernel : Kernel features
    posterior_kernel x x' = kernel x x' - (@@) {head=[_]} ((@@) {head=[_]} (kernel x x_train) inv) (kernel x_train x')

export
marginalise : {samples : Nat} -> GaussianProcess features -> Tensor (samples :: features) Double -> Gaussian samples []
marginalise (MkGP mean_function kernel) x = MkGaussian (mean_function x) (kernel x x)

PI : Double

log_marginal_likelihood : {samples : Nat}
 -> GaussianProcess features
 -> Gaussian (S samples) []
 -> (Tensor ((S samples) :: features) Double, Tensor [S samples] Double)
 -> Maybe $ Tensor [] Double
log_marginal_likelihood (MkGP _ kernel) (MkGaussian _ cov) (x, y) = map foo $ inverse (kernel x x + cov) where
  foo : Tensor [S samples, S samples] Double -> Tensor [] Double
  foo inv = let a = (@@) {head=[]} ((@@) {head=[]} y inv) y
                b = (log $ det inv)
                c = (MkTensor $ the Double $ cast samples) * (log $ MkTensor $ 2.0 * PI) in
                  (MkTensor (-1.0 / 2)) * (a - b + c)

export
optimize : {samples : Nat}
 -> Optimizer hp
 -> (hp -> GaussianProcess features)
 -> Gaussian (S samples) []
 -> (Tensor ((S samples) :: features) Double, Tensor [S samples] Double)
 -> Maybe hp
optimize optimizer gp_from_hyperparameters likelihood training_data = optimizer objective where
  objective : hp -> Maybe $ Tensor [] Double
  objective hp' = log_marginal_likelihood (gp_from_hyperparameters hp') likelihood training_data
