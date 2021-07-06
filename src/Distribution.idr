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
import Tensor
import Util

||| A joint, or multivariate distribution over a tensor of floating point values, where the first
||| two central moments (mean and covariance) are known. Every sub-event is assumed to have the
||| same shape.
public export
interface TwoCentralMoment (0 event_shape : Shape) (0 dist : Nat -> Type) where
  ||| The mean of the distribution.
  mean : dist dim -> Tensor (dim :: event_shape) Double

  ||| The covariance, or correlation, between sub-events.
  covariance : dist dim -> Tensor (dim :: dim :: event_shape) Double

||| The variance of a single random variable.
export
variance : TwoCentralMoment event_shape dist => dist 1 -> Tensor (1 :: event_shape) Double
variance dist = squeeze {from=(1 :: 1 :: event_shape)} $ covariance dist

||| A joint, or multivariate distribution over a tensor of floating point values, where the density
||| function and corresponding cumulative density function are known (either analytically or via
||| approximation). Every sub-event is assumed to have the same shape.
public export
interface Distribution (0 event_shape : Shape) (0 dist : Nat -> Type) where
  ||| The probability density function of the distribution at the specified point.
  pdf : dist (S d) -> Tensor (S d :: event_shape) Double -> Tensor [] Double

  ||| The cumulative distribution function of the distribution at the specified point (that is, the
  ||| probability the random variable takes a value less than or equal to the given point).
  cdf : dist (S d) -> Tensor (S d :: event_shape) Double -> Tensor [] Double

||| A joint Gaussian distribution.
public export
data Gaussian : (0 event_shape : Shape) -> (dim : Nat) -> Type where
  ||| @mean The Gaussian mean.
  ||| @covariance The Gaussian covariance.
  MkGaussian : {d : Nat} -> (mean : Tensor (S d :: event_shape) Double) ->
               (covariance : Tensor (S d :: S d :: event_shape) Double) ->
               Gaussian event_shape (S d)

export
TwoCentralMoment e (Gaussian e) where
  mean (MkGaussian mean' _) = mean'
  covariance  (MkGaussian _ cov) = cov

export
Distribution [1] (Gaussian [1]) where
  pdf (MkGaussian {d} mean cov) x =
    let diff : Tensor [S d, 1] Double
        diff = x - mean

        exponent : Tensor [] Double
        exponent = - (squeeze $ diff.T @@ cov.T @@ diff) / (const {shape=[]} 2.0)

        denominator : Tensor [] Double
        denominator = (const {shape=[]} $ 2 * PI) ^ (const {shape=[]} $ cast (S d) / 2.0)
                      * (det $ squeeze {to=[S d, S d]} cov) ^ (const {shape=[]} 0.5)

     in (exp exponent) / denominator

  cdf (MkGaussian mean cov) x = ?cdf_rhs
