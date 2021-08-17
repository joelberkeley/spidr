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

||| A joint, or multivariate distribution over a tensor of floating point values, where the first
||| two central moments (mean and covariance) are known. Every sub-event is assumed to have the
||| same shape.
|||
||| @event The shape of each sub-event.
||| @dist Constructs the distribution from the number of events in the distribution.
public export
interface Distribution (0 event : Shape) (0 dist : (0 dim : Nat) -> Type) | dist where
  ||| The mean of the distribution.
  mean : dist dim -> Tensor (dim :: event) Double

  ||| The covariance, or correlation, between sub-events.
  cov : dist dim -> Tensor (dim :: dim :: event) Double

||| The variance of a single random variable.
export
variance : Distribution event dist => dist 1 -> Tensor (1 :: event) Double
variance dist = squeeze {from=(1 :: 1 :: event)} $ cov dist

||| A joint, or multivariate distribution over a tensor of floating point values, where the density
||| function and corresponding cumulative density function are known (either analytically or via
||| approximation). Every sub-event is assumed to have the same shape.
|||
||| @event The shape of each sub-event.
||| @dist Constructs the distribution from the number of events in the distribution.
public export
interface Distribution event dist =>
  ClosedFormDistribution (0 event : Shape) (0 dist : (0 dim : Nat) -> Type) where
    ||| The probability density function of the distribution at the specified point.
    pdf : dist (S d) -> Tensor (S d :: event) Double -> Tensor [] Double

    ||| The cumulative distribution function of the distribution at the specified point (that is,
    ||| the probability the random variable takes a value less than or equal to the given point).
    cdf : dist (S d) -> Tensor (S d :: event) Double -> Tensor [] Double

||| A joint Gaussian distribution.
|||
||| @event The shape of each sub-event.
||| @dim The number of sub-events.
public export
data Gaussian : (0 event : Shape) -> (0 dim : Nat) -> Type where
  ||| @mean The mean of the events.
  ||| @cov The covariance between events.
  MkGaussian : {d : Nat} -> (mean : Tensor (S d :: event) Double) ->
               (cov : Tensor (S d :: S d :: event) Double) ->
               Gaussian event (S d)

export
Distribution event (Gaussian event) where
  mean (MkGaussian mean' _) = mean'
  cov (MkGaussian _ cov') = cov'

export
ClosedFormDistribution [1] (Gaussian [1]) where
  pdf (MkGaussian {d} mean cov) x =
    let diff : Tensor [S d, 1] Double
        diff = x - mean

        exponent : Tensor [] Double
        exponent = - (squeeze $ diff.T @@ cov.T @@ diff) / (const {shape=[]} 2.0)

        denominator : Tensor [] Double
        denominator = (const {shape=[]} $ 2 * pi) ^ (const {shape=[]} $ cast (S d) / 2.0)
                      * (det $ squeeze {to=[S d, S d]} cov) ^ (const {shape=[]} 0.5)

     in (exp exponent) / denominator

  cdf (MkGaussian mean cov) x = ?cdf_rhs
