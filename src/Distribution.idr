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
||| This module contains definitions and implementations for probability distributions.
module Distribution

import Tensor

||| A joint, or multivariate distribution over a tensor of floating point values. Every sub-event
||| is assumed to have the same shape.
public export
-- todo should we make dim implicit?
interface Distribution (0 event_shape : Shape) (0 dist : Nat -> Type) where
  ||| The mean of the distribution
  mean : dist dim -> Tensor (dim :: event_shape) Double

  ||| The covariance, or correlation between sub-events
  covariance : dist dim -> Tensor (dim :: dim :: event_shape) Double

-- todo should we squeeze the first dim on the output?
||| The variance of a single random variable
export
variance : Distribution event_shape dist => dist 1 -> Tensor (1 :: event_shape) Double

||| A joint Gaussian distribution.
public export
data Gaussian : (0 event_shape : Shape) -> (dim : Nat) -> Type where
  ||| @mean The Gaussian mean.
  ||| @covariance The Gaussian covariance.
  MkGaussian : (mean : Tensor (dim :: event_shape) Double) ->
               (covariance : Tensor (dim :: dim :: event_shape) Double) ->
               Gaussian event_shape dim

export
Distribution e (Gaussian e) where
  mean (MkGaussian mean' _) = mean'
  covariance  (MkGaussian _ cov) = cov

||| The probability density function of the Gaussian at the specified point.
export
pdf : Gaussian event_shape dim -> Tensor (dim :: event_shape) Double -> Tensor [] Double

||| The cumulative distribution function of the Gaussian at the specified point (that is, the
||| probability the random variable takes a value less than or equal to the given point).
export
cdf : Gaussian event_shape 1 -> Tensor (1 :: event_shape) Double -> Tensor [] Double
