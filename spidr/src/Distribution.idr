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
||| This module contains definitions for probability distributions.
module Distribution

import Data.Nat
import Literal
import Tensor
import Constants

||| A joint, or multivariate distribution over a tensor of floating point values, where the first
||| two central moments (mean and covariance) are known. Every sub-event is assumed to have the
||| same shape.
|||
||| @dist Constructs the distribution from the shape of each sub-event and the number of events in
|||   the distribution.
public export
interface Distribution (0 dist : (0 event : Shape) -> (0 dim : Nat) -> Type) where
  ||| The mean of the distribution.
  mean : dist event dim -> Graph $ Tensor (dim :: event) F64

  ||| The covariance, or correlation, between sub-events.
  cov : dist event dim -> Graph $ Tensor (dim :: dim :: event) F64

||| The variance of a single random variable.
export
variance : {event : _} -> Distribution dist => dist event 1 -> Graph $ Tensor (1 :: event) F64
variance dist = squeeze {from = (1 :: 1 :: event)} =<< cov dist

||| A joint, or multivariate distribution over a tensor of floating point values, where the density
||| function and corresponding cumulative density function are known (either analytically or via
||| approximation). Every sub-event is assumed to have the same shape.
|||
||| @event The shape of each sub-event.
||| @dist Constructs the distribution from the shape of each sub-event and the number of events in
|||   the distribution.
public export
interface Distribution dist  =>
  ClosedFormDistribution (0 event : Shape)
    (0 dist : (0 event : Shape) -> (0 dim : Nat) -> Type) where
      ||| The probability density function of the distribution at the specified point.
      pdf : dist event (S d) -> Tensor (S d :: event) F64 -> Graph $ Tensor [] F64

      ||| The cumulative distribution function of the distribution at the specified point (that is,
      ||| the probability the random variable takes a value less than or equal to the given point).
      cdf : dist event (S d) -> Tensor (S d :: event) F64 -> Graph $ Tensor [] F64

||| A joint Gaussian distribution.
|||
||| @event The shape of each sub-event.
||| @dim The number of sub-events.
public export
data Gaussian : (0 event : Shape) -> (0 dim : Nat) -> Type where
  ||| @mean The mean of the events.
  ||| @cov The covariance between events.
  MkGaussian : {d : Nat} -> (mean : Tensor (S d :: event) F64) ->
               (cov : Tensor (S d :: S d :: event) F64) ->
               Gaussian event (S d)

export
Distribution Gaussian where
  mean (MkGaussian mean' _) = pure mean'
  cov (MkGaussian _ cov') = pure cov'

||| **NOTE** `cdf` is implemented only for univariate `Gaussian`.
export
ClosedFormDistribution [1] Gaussian where
  pdf (MkGaussian {d} mean cov) x = do
    cholCov <- cholesky =<< squeeze {to = [S d, S d]} cov
    tri <- pure cholCov |\ squeeze !(pure x - pure mean)
    let exponent = - pure tri @@ pure tri / 2.0
        covSqrtDet = reduce @{Prod} [0] !(diag cholCov)
        denominator = (fromDouble $ pow (2.0 * pi) (cast (S d) / 2.0)) * covSqrtDet
    (exp !exponent) / denominator

  cdf (MkGaussian {d = S _} _ _) _ = ?multivariate_cdf
  cdf (MkGaussian {d = 0} mean cov) x =
    (1.0 + erf !(squeeze !(pure x - pure mean) / (sqrt !(squeeze cov * 2.0)))) / 2.0
